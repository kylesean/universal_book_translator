const { chromium } = require('playwright');

const BASE_URL = process.env.BASE_URL || 'http://127.0.0.1:8000';

async function testExportAndReadingFlow() {
  console.log('=== TESTING ONE-CLICK EXPORT & READING FLOW IN REAL BROWSER ===');
  const browser = await chromium.launch({
    headless: true,
    executablePath: '/usr/bin/google-chrome-stable',
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage']
  });

  const context = await browser.newContext({
    viewport: { width: 1440, height: 900 },
    locale: 'zh-CN',
    acceptDownloads: true,
  });

  const page = await context.newPage();
  await page.addInitScript(() => {
    try {
      localStorage.setItem('ubt_lang', 'zh');
    } catch {
      // Ignore local storage error in sandboxed context
    }
  });

  const consoleErrors = [];
  const uncaughtExceptions = [];

  page.on('response', (res) => {
    if (res.status() >= 400) {
      console.log(`[HTTP ${res.status()}] ${res.url()}`);
    }
  });

  page.on('console', (msg) => {
    if (msg.type() === 'error') {
      consoleErrors.push(`[Console Error] ${msg.text()}`);
    }
  });

  page.on('pageerror', (err) => {
    uncaughtExceptions.push(`[Uncaught Page Error] ${err.message}`);
  });

  let testsPassed = 0;
  let testsFailed = 0;

  function record(testName, pass, detail = '') {
    if (pass) {
      testsPassed++;
      console.log(`✅ PASS: ${testName} ${detail ? '(' + detail + ')' : ''}`);
    } else {
      testsFailed++;
      console.error(`❌ FAIL: ${testName} ${detail ? '(' + detail + ')' : ''}`);
    }
  }

  try {
    // -------------------------------------------------------------
    // Test 1: Mission Control - One Click Export & Format Dropdown
    // -------------------------------------------------------------
    console.log('\n--- 1. Mission Control One-Click Export ---');
    await page.goto(`${BASE_URL}/jobs`, { waitUntil: 'networkidle', timeout: 15000 });
    await page.waitForTimeout(1000);

    const jobRow = page.locator('tr:has-text("web_mvnvgck2")').first();
    await jobRow.click();
    await page.waitForTimeout(1000);

    // Check One-Click Export button
    const exportBtn = page.locator('button:has-text("一键导出交付"), button:has-text("One-Click Export")').first();
    record('Mission Control export button visible', await exportBtn.isVisible());

    // Check dropdown toggle button
    const dropdownToggle = page.locator('button[title*="交付格式"], button[title*="format"]').first();
    record('Mission Control format dropdown toggle exists', await dropdownToggle.isVisible());

    if (await dropdownToggle.isVisible()) {
      const exportBox = await exportBtn.boundingBox();
      const toggleBox = await dropdownToggle.boundingBox();
      record('Mission Control button and toggle heights match', exportBox && toggleBox && Math.abs(exportBox.height - toggleBox.height) <= 1,
        `export h=${exportBox?.height}, toggle h=${toggleBox?.height}`);

      await dropdownToggle.click();
      await page.waitForTimeout(400);

      const dropdownMenu = page.locator('.shadow-lg button').first();
      record('Format dropdown menu opened', await dropdownMenu.isVisible());

      const epubOption = page.locator('.shadow-lg button').filter({ hasText: /epub/i }).first();
      record('EPUB deliverable option visible in menu', await epubOption.isVisible());

      const localizedPrimary = page.locator('.shadow-lg button').filter({ hasText: /高保真|Target PDF/i }).first();
      record('Primary deliverable localized name displayed', await localizedPrimary.isVisible());

      // Click toggle again to dismiss
      await dropdownToggle.click();
      await page.waitForTimeout(300);
      record('Dropdown closes on toggle click', !(await page.locator('.shadow-lg').isVisible()));
    }

    // -------------------------------------------------------------
    // Test 2: Review Workbench - One Click Export & Format Dropdown
    // -------------------------------------------------------------
    console.log('\n--- 2. Review Workbench One-Click Export ---');
    await page.goto(`${BASE_URL}/jobs/web_mvnvgck2_baae68e98c68/review`, { waitUntil: 'networkidle', timeout: 15000 });
    await page.waitForTimeout(2000);

    const wbExportBtn = page.locator('button:has-text("一键导出交付"), button:has-text("One-Click Export")').first();
    record('Review Workbench One-Click Export button visible', await wbExportBtn.isVisible());

    const wbFormatToggle = page.locator('button[title*="交付格式"], button[title*="format"]').first();
    record('Review Workbench format dropdown toggle visible', await wbFormatToggle.isVisible());

    if (await wbFormatToggle.isVisible()) {
      const wbExportBox = await wbExportBtn.boundingBox();
      const wbToggleBox = await wbFormatToggle.boundingBox();
      record('Review Workbench button and toggle heights match', wbExportBox && wbToggleBox && Math.abs(wbExportBox.height - wbToggleBox.height) <= 1,
        `export h=${wbExportBox?.height}, toggle h=${wbToggleBox?.height}`);

      await wbFormatToggle.click();
      await page.waitForTimeout(400);

      const wbDropdownMenu = page.locator('.shadow-lg button').first();
      record('Review Workbench format menu opens', await wbDropdownMenu.isVisible());

      const htmlOption = page.locator('.shadow-lg button').filter({ hasText: /html/i }).first();
      record('HTML deliverable option visible in workbench menu', await htmlOption.isVisible());

      const localizedHtml = page.locator('.shadow-lg button').filter({ hasText: /可读 HTML|Readable HTML/i }).first();
      record('HTML deliverable localized name displayed', await localizedHtml.isVisible());

      // Click toggle again to dismiss
      await wbFormatToggle.click();
      await page.waitForTimeout(300);
      record('Workbench dropdown closes on toggle click', !(await page.locator('.shadow-lg').isVisible()));
    }

    // -------------------------------------------------------------
    // Test 3: Reading Flow Virtualization & Editing
    // -------------------------------------------------------------
    console.log('\n--- 3. Reading Flow Dynamic Measurement & Virtualization ---');
    const flowTab = page.locator('button').filter({ hasText: /通读|Flow/i }).first();
    record('Reading Flow tab button found', await flowTab.isVisible());
    await flowTab.click();
    await page.waitForTimeout(1000);

    const flowItems = page.locator('div[data-index]');
    const initialFlowCount = await flowItems.count();
    record('Reading flow virtual items mounted', initialFlowCount > 0, `Found ${initialFlowCount} items in view`);

    // Measure bounding boxes of the first two items
    const firstBox = await flowItems.nth(0).boundingBox();
    const secondBox = await flowItems.nth(1).boundingBox();
    record('Virtual items positioned correctly without overlap', firstBox && secondBox && secondBox.y >= firstBox.y + firstBox.height - 2,
      `Item 0: y=${firstBox?.y}, h=${firstBox?.height}; Item 1: y=${secondBox?.y}, h=${secondBox?.height}`);

    // Test inline edit double-click in Reading Flow
    const firstFlowItem = flowItems.first();
    await firstFlowItem.click();
    await page.waitForTimeout(400);

    const firstTargetPara = firstFlowItem.locator('p.select-text').first();
    await firstTargetPara.dblclick();
    await page.waitForTimeout(400);

    const textarea = firstFlowItem.locator('textarea').first();
    record('Reading Flow inline textarea appears on double-click', await textarea.isVisible());

    if (await textarea.isVisible()) {
      const hint = firstFlowItem.locator('span.font-mono').first();
      const hintText = await hint.textContent();
      record('Reading Flow shortcut hint localized correctly', hintText && hintText.includes('Ctrl+Enter') && (hintText.includes('保存') || hintText.includes('save')), `Hint: ${hintText}`);

      // Type an edit into the textarea without pressing Ctrl+Enter
      await textarea.fill('Auto-saved draft test ' + Date.now());
      await page.waitForTimeout(300);

      // Trigger One-Click Export: should auto-save dirty draft before downloading
      await wbExportBtn.click();
      await page.waitForTimeout(1000);

      const busyOrToast = page.locator('button .animate-spin, div:has-text("保存"), div:has-text("渲染"), div:has-text("下载")').first();
      record('One-Click Export flushes unsaved draft into export pipeline', await busyOrToast.isVisible());

      // Wait for re-render to complete
      await page.waitForTimeout(4500);
      record('Inline textarea unmounted after auto-save and export', !(await textarea.isVisible()));
    }

    // -------------------------------------------------------------
    // Test 4: Switching between Reading Flow and Segment Grid
    // -------------------------------------------------------------
    console.log('\n--- 4. View Mode Switching & Layout Stability ---');
    // Select block in reading flow
    if (initialFlowCount >= 4) {
      await flowItems.nth(3).click();
      await page.waitForTimeout(400);
      const inspector = page.locator('aside').first();
      record('Clicking item opens Inspector drawer in Reading Flow', await inspector.isVisible());
    }

    // Switch back to Segments
    await page.locator('button:has-text("分块对齐网格"), button:has-text("Segment")').first().click();
    await page.waitForTimeout(600);
    const segItems = page.locator('div.pb-4 div.cursor-pointer');
    record('Switched back to Segment Grid without error', (await segItems.count()) > 0);

    // Switch back to Reading Flow
    await flowTab.click();
    await page.waitForTimeout(600);
    record('Switched back to Reading Flow smoothly', (await page.locator('div[data-index]').count()) > 0);

    // -------------------------------------------------------------
    // Test 5: Check Uncaught Page Exceptions & Console Errors
    // -------------------------------------------------------------
    console.log('\n--- 5. Runtime Exceptions & Console Integrity ---');
    record('Zero uncaught page errors', uncaughtExceptions.length === 0, uncaughtExceptions.join('; '));
    record('Zero console errors', consoleErrors.length === 0, consoleErrors.join('; '));

  } catch (err) {
    console.error('Fatal test error:', err);
    testsFailed++;
  } finally {
    await browser.close();
  }

  console.log(`\n=== TEST RUN FINISHED: ${testsPassed} passed, ${testsFailed} failed ===`);
  process.exit(testsFailed > 0 ? 1 : 0);
}

testExportAndReadingFlow();
