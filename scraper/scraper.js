const puppeteer = require("puppeteer");
const fs = require("fs");

(async () => {
  const browser = await puppeteer.launch({ headless: "new" });
  const page = await browser.newPage();

  page.goto("https://www.chess.com/games/view/16074142");

  const downloadButton = await page.waitForSelector(
    ".secondary-controls-button"
  );

  await downloadButton.click();

  const dataSel = await page.waitForFunction(
    "document.querySelectorAll('.share-menu-tab-pgn-section')[1].childNodes[2].value"
  );

  const data = await dataSel.evaluate((el) => el);
  await fs.appendFile("data.txt", `\n\n${data}`, (err) => {
    if (err) throw err;
  });

  await browser.close();
})();
