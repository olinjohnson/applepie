const puppeteer = require("puppeteer");
const fs = require("fs");

(async () => {
  const browser = await puppeteer.launch({ headless: "new" });
  const page = await browser.newPage();
  await page.goto(`https://www.chess.com/games/garry-kasparov`);

  let c2 = 1;
  const game_elems = await page.$$(".master-games-master-game");

  await game_elems.forEach((element) => {
    console.log(`hello ${c2}`);
    console.log(element.$("td")); //.$("a").click();
    c2++;
  });

  /*page.goto(
    `https://www.chess.com/games/search?fromSearchShort=1&p1=Garry%20Kasparov&playerId=21779&page=${counter}`
  );*/

  const test = await page.$("asdkjhasdkjhf");
  console.log(`hi ${test}`);

  await browser.close();
})();

/*
(async () => {
  const browser = await puppeteer.launch({ headless: "new" });
  const page = await browser.newPage();

  await page.goto("https://www.chess.com/games/view/16074142");

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
*/
