// ======================= 侧边栏面板切换 =======================

const navItems = document.querySelectorAll(".nav-item[data-tab]");
const panels = document.querySelectorAll(".tab-panel");
const gotoButtons = document.querySelectorAll("[data-goto]");

function activateTab(tabId) {
  navItems.forEach((item) => {
    item.classList.toggle("active", item.dataset.tab === tabId);
  });
  panels.forEach((panel) => {
    panel.classList.toggle("active", panel.id === "tab-" + tabId);
  });

  // 保持其它查询参数不变，仅更新 tab，便于刷新后仍停留在当前面板。
  const url = new URL(location.href);
  url.searchParams.set("tab", tabId);
  history.replaceState(null, "", url);

  window.scrollTo({ top: 0, behavior: "smooth" });
}

navItems.forEach((item) => {
  item.addEventListener("click", () => activateTab(item.dataset.tab));
});

gotoButtons.forEach((el) => {
  el.addEventListener("click", () => activateTab(el.dataset.goto));
});

// 初始面板：优先读取 URL 中的 tab 参数，其次根据内容自动选择。
const params = new URLSearchParams(location.search);
const validTabs = Array.from(navItems).map((item) => item.dataset.tab);
let initialTab = params.get("tab");

if (!initialTab || !validTabs.includes(initialTab)) {
  if (params.get("shp_preview") && !params.get("annotation")) {
    initialTab = "shp";
  } else if (params.get("annotation")) {
    initialTab = "annotate";
  } else {
    initialTab = "overview";
  }
}

activateTab(initialTab);

// ======================= 叠加图点击记录坐标 =======================

const overlayImage = document.getElementById("overlay-image");
const pointX = document.getElementById("point-x");
const pointY = document.getElementById("point-y");

if (overlayImage && pointX && pointY) {
  overlayImage.addEventListener("click", (event) => {
    const rect = overlayImage.getBoundingClientRect();
    const scaleX = overlayImage.naturalWidth / rect.width;
    const scaleY = overlayImage.naturalHeight / rect.height;
    const x = Math.round((event.clientX - rect.left) * scaleX);
    const y = Math.round((event.clientY - rect.top) * scaleY);
    pointX.value = x;
    pointY.value = y;
  });
}

// ======================= Shp 记录与几何联动高亮 =======================

const shpRecordRows = document.querySelectorAll(".shp-record-row");
const shpElements = document.querySelectorAll(".shp-path, .shp-point");

function setActiveShpRecord(recordIndex) {
  shpRecordRows.forEach((row) => {
    row.classList.toggle("active", row.dataset.recordIndex === recordIndex);
  });
  shpElements.forEach((element) => {
    element.classList.toggle("active", element.dataset.recordIndex === recordIndex);
  });
}

if (shpRecordRows.length && shpElements.length) {
  shpRecordRows.forEach((row) => {
    row.addEventListener("mouseenter", () => setActiveShpRecord(row.dataset.recordIndex));
    row.addEventListener("click", () => setActiveShpRecord(row.dataset.recordIndex));
  });
  shpElements.forEach((element) => {
    element.addEventListener("mouseenter", () => setActiveShpRecord(element.dataset.recordIndex));
    element.addEventListener("click", () => setActiveShpRecord(element.dataset.recordIndex));
  });
}

// ======================= 主体分类类别筛选联动 =======================

const categoryFilters = document.querySelectorAll(".category-filter");
const instanceBoxes = document.querySelectorAll(".instance-box");
const instanceRows = document.querySelectorAll(".instance-row");

function applyCategoryFilter(category) {
  categoryFilters.forEach((button) => {
    button.classList.toggle("active", button.dataset.category === category);
  });

  instanceBoxes.forEach((box) => {
    const matched = category === "all" || box.dataset.category === category;
    box.classList.toggle("active", matched && category !== "all");
    box.classList.toggle("dimmed", !matched && category !== "all");
  });

  instanceRows.forEach((row) => {
    const matched = category === "all" || row.dataset.category === category;
    row.classList.toggle("active", matched && category !== "all");
    row.classList.toggle("dimmed", !matched && category !== "all");
  });
}

if (categoryFilters.length) {
  categoryFilters.forEach((button) => {
    button.addEventListener("click", () => {
      applyCategoryFilter(button.dataset.category || "all");
      // 在「主体分类」页点击某个类别后，跳转到标注编辑页查看对应实例。
      if (button.dataset.jumpTab) {
        activateTab(button.dataset.jumpTab);
      }
    });
  });
  applyCategoryFilter("all");
}
