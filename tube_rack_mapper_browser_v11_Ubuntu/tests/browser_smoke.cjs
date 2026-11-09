/* Run with NODE_PATH pointing to a Playwright installation. Uses a simulated camera.
   SCANNER_URL defaults to http://localhost:18525/matrix; CHROME_BIN is optional. */
const {chromium} = require('playwright');
const assert = require('node:assert/strict');
const path = require('node:path');
const fs = require('node:fs');
const executablePath = process.env.CHROME_BIN || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const url = process.env.SCANNER_URL || 'http://localhost:18525/matrix';
(async () => {
  const browser = await chromium.launch({executablePath, headless: true, args: ['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream']});
  try {
    const page = await browser.newPage({viewport: {width: 1440, height: 1200}, permissions: ['camera']});
    page.setDefaultTimeout(30000);
    await page.goto(url);
    await page.getByText('Webcam', {exact: true}).click();
    const camera = page.frameLocator('iframe[title$="built_rack_camera"]');
    await camera.getByRole('button', {name: 'Enable camera'}).click();
    await camera.getByRole('button', {name: 'Capture in 3 seconds'}).waitFor();
    assert.equal(await page.locator('.stApp').evaluate(e => getComputedStyle(e).backgroundColor), 'rgb(255, 255, 255)');
    await camera.getByRole('button', {name: 'Full screen light'}).click();
    await camera.getByRole('button', {name: 'Leave full screen'}).click();
    await camera.locator('#stage').focus();
    await page.keyboard.press('Space');
    await camera.getByRole('button', {name: 'Cancel countdown'}).click();
    await camera.getByRole('button', {name: 'Capture in 3 seconds'}).click();
    await camera.getByRole('button', {name: 'Use this photo'}).waitFor();
    // The camera must be off while reviewing a still.
    assert.equal(await camera.locator('#video').evaluate(e => e.srcObject), null);
    await camera.getByRole('button', {name: 'Use this photo'}).click();
    await page.waitForFunction(() => {const b = document.querySelector('[data-testid="stBaseButton-primary"]'); return b && !b.disabled;});
    assert.equal(await page.getByRole('button', {name: 'Start analysis', exact: true}).isEnabled(), true);
    await camera.getByRole('button', {name: 'Retake', exact: true}).click();
    await camera.getByRole('button', {name: 'Capture in 3 seconds'}).waitFor();
    await page.waitForFunction(() => document.querySelector('[data-testid="stBaseButton-primary"]')?.disabled);
    await camera.getByRole('button', {name: 'Exit camera'}).click();
    await page.locator('input[type=file]').waitFor({state: 'attached'});
    assert.equal(await page.locator('iframe').count(), 0);
    console.log('PASS: white lighting, fullscreen, keyboard, countdown, capture, accept, retake, exit');

    await page.getByText('Webcam', {exact: true}).click();
    await camera.locator('#status').waitFor();
    await camera.locator('body').evaluate(() => {navigator.mediaDevices.getUserMedia = () => Promise.reject(new DOMException('denied', 'NotAllowedError'));});
    await camera.getByRole('button', {name: 'Enable camera'}).click();
    await camera.locator('#status.error').waitFor();
    assert.match(await camera.locator('#status').innerText(), /Camera access was blocked/);
    await camera.getByRole('button', {name: 'Exit camera'}).click();
    await page.locator('input[type=file]').waitFor({state: 'attached'});
    console.log('PASS: denied camera permission with upload fallback');

    await page.locator('input[type=file]').setInputFiles(path.resolve(__dirname, '../sample_rack.jpg'));
    await page.getByRole('button', {name: 'Start analysis', exact: true}).click();
    await page.getByRole('button', {name: 'Download plate-map CSV'}).waitFor({timeout: 180000});
    const [download] = await Promise.all([page.waitForEvent('download'), page.getByRole('button', {name: 'Download plate-map CSV'}).click()]);
    const csv = fs.readFileSync(await download.path(), 'utf8');
    assert.equal(csv.trim().split('\n').length, 97);
    assert.equal(csv.split('\n')[0].trim(), 'Plate,Position,DataMatrix');
    await page.getByRole('textbox', {name: 'Plate ID override (optional)'}).fill('NEW-PLATE');
    await page.getByRole('textbox', {name: 'Plate ID override (optional)'}).press('Enter');
    await page.getByRole('button', {name: 'Download plate-map CSV'}).waitFor({state: 'hidden'});
    assert.equal(await page.getByTestId('stException').count(), 0);
    console.log('PASS: real sample decode, 96-well CSV, stale export invalidation');
  } finally {await browser.close();}
})().catch(error => {console.error(error); process.exitCode = 1;});
