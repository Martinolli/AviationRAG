import { expect, Page, test } from "@playwright/test";

function buildSessionList(count: number) {
  return Array.from({ length: count }, (_, index) => ({
    id: `session-${String(index + 1).padStart(3, "0")}`,
    title: `Session ${String(index + 1).padStart(3, "0")}`,
    created_at: `2026-02-01T10:${String(index).padStart(2, "0")}:00.000Z`,
    updated_at: `2026-02-01T11:${String(index).padStart(2, "0")}:00.000Z`,
    pinned: index < 2,
  }));
}

function buildHistoryRows(count: number) {
  return Array.from({ length: count }, (_, index) => ({
    timestamp: `2026-02-01T12:${String(index).padStart(2, "0")}:00.000Z`,
    user_query: `User question ${index + 1}`,
    ai_response: `Assistant answer ${index + 1} with practical guidance.`,
  }));
}

async function mockShellApis(page: Page) {
  await page.route("**/api/auth/session", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        user: { email: "smoke@test.dev" },
        expires: "2099-12-31T23:59:59.999Z",
      }),
    });
  });

  await page.route("**/api/health", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ success: true }),
    });
  });

  await page.route("**/api/chat/session?*", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        success: true,
        sessions: buildSessionList(120),
      }),
    });
  });

  await page.route("**/api/chat/history/*", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        success: true,
        messages: buildHistoryRows(80),
      }),
    });
  });

  await page.route("**/api/chat/ask", async (route) => {
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        success: true,
        answer: "Mock assistant answer with citation.",
        citations: [{ filename: "mock-regulation.pdf", chunk_id: "chunk-01" }],
        sources: [
          {
            filename: "mock-regulation.pdf",
            chunk_id: "chunk-01",
            text: "Mock source passage used by smoke test.",
          },
        ],
      }),
    });
  });
}

test("chat shell keeps composer visible and drawer overlays without shrinking chat", async ({ page }) => {
  await mockShellApis(page);
  await page.goto("/");

  const chatPanel = page.getByTestId("chat-panel");
  const composer = page.getByTestId("composer");
  const sidebarList = page.getByTestId("session-list");
  const messageArea = page.getByTestId("message-area");
  const sourceDrawer = page.getByTestId("source-drawer");

  await expect(chatPanel).toBeVisible();
  await expect(composer).toBeVisible();

  const sidebarBefore = await sidebarList.evaluate((element) => element.scrollTop);
  await sidebarList.evaluate((element) => {
    element.scrollTop = element.scrollHeight;
  });
  const sidebarAfter = await sidebarList.evaluate((element) => element.scrollTop);
  expect(sidebarAfter).toBeGreaterThan(sidebarBefore);
  await expect(composer).toBeVisible();

  await page.getByRole("button", { name: "Session 001" }).click();
  const messageBefore = await messageArea.evaluate((element) => element.scrollTop);
  await messageArea.evaluate((element) => {
    element.scrollTop = element.scrollHeight;
  });
  const messageAfter = await messageArea.evaluate((element) => element.scrollTop);
  expect(messageAfter).toBeGreaterThan(messageBefore);
  await expect(composer).toBeVisible();

  const chatWidthBefore = await chatPanel.evaluate((element) => element.getBoundingClientRect().width);
  await page.getByRole("button", { name: "Show Sources" }).click();
  await expect(sourceDrawer).toHaveClass(/drawerOpen/);
  const chatWidthAfter = await chatPanel.evaluate((element) => element.getBoundingClientRect().width);
  expect(Math.abs(chatWidthAfter - chatWidthBefore)).toBeLessThanOrEqual(1);
  await sourceDrawer.getByRole("button", { name: "Close" }).click();
  await expect(sourceDrawer).not.toHaveClass(/drawerOpen/);

  await page.getByTestId("composer").locator("textarea").fill("Please provide a cited answer.");
  await page.getByRole("button", { name: "Send" }).click();
  await expect(page.getByTestId("citation-chip").first()).toBeVisible();
  await page.getByTestId("citation-chip").first().click();
  await expect(sourceDrawer).toHaveClass(/drawerOpen/);
  await expect(page.getByText("Mock source passage used by smoke test.")).toBeVisible();

  const pageScrollable = await page.evaluate(() => {
    const scrolling = document.scrollingElement;
    return Boolean(scrolling && scrolling.scrollHeight > scrolling.clientHeight + 1);
  });
  expect(pageScrollable).toBeFalsy();
});
