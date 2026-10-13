// Capture the actual documentation evidence page in Chromium.
const fs = require('node:fs');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const { chromium } = require('playwright');

async function main() {
  const options = { headless: true };
  if (process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE) {
    options.executablePath = process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE;
  }
  const browser = await chromium.launch(options);
  try {
    const page = await browser.newPage({
      viewport: { width: 1280, height: 960 }, deviceScaleFactor: 1,
      colorScheme: 'light', reducedMotion: 'reduce',
    });
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(pathToFileURL(path.join(__dirname,
      'evidence/qemu-2026-10-13/index.html')).href);
    await page.evaluate(() => document.fonts.ready);
    if (await page.locator('.card').count() !== 2) {
      throw new Error('Both evidence cards must render before capture');
    }
    for (const width of [390, 768, 1280]) {
      await page.setViewportSize({ width, height: 960 });
      if (await page.evaluate(() =>
        document.documentElement.scrollWidth > window.innerWidth)) {
        throw new Error(`Horizontal overflow at ${width}px`);
      }
    }
    if (errors.length) throw new Error(errors.join('\n'));
    await page.screenshot({ path: path.join(__dirname,
      'assets/evidence-screenshot.png'), fullPage: true });
    fs.writeFileSync(path.join(__dirname, 'assets/capture.json'),
      JSON.stringify({ source: 'docs/evidence/qemu-2026-10-13/index.html',
        viewport: { width: 1280, height: 960 }, device_scale_factor: 1,
        playwright: require('playwright/package.json').version,
        chromium: browser.version(), full_page: true,
        horizontal_overflow_checked_at_widths: [390, 768, 1280],
      }, null, 2) + '\n');
    console.log('Captured evidence page and checked desktop/mobile overflow.');
  } finally {
    await browser.close();
  }
}

main().catch(error => { console.error(error); process.exitCode = 1; });
