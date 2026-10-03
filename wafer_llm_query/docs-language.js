(() => {
  const root = document.getElementById('swagger-ui');
  const select = document.getElementById('docs-language');
  const pairs = [
    ['Wafer React Query API', '晶圓查詢 API'],
    ['default', 'API 端點'], ['Health', '健康檢查'],
    ['Overview', '資料概況'], ['Query', '自然語言查詢'],
    ['Try it out', '試用'], ['Execute', '執行'], ['Cancel', '取消'],
    ['Clear', '清除'], ['Reset', '重設'], ['Download', '下載'],
    ['Parameters', '參數'], ['No parameters', '無參數'],
    ['Name', '名稱'], ['Description', '說明'], ['Required', '必填'], ['required', '必填'],
    ['Request body', '請求內容'], ['Request URL', '請求網址'],
    ['Responses', '回應'], ['Server response', '伺服器回應'],
    ['Response body', '回應內容'], ['Response headers', '回應標頭'],
    ['Code', '狀態碼'], ['Details', '詳細內容'], ['Links', '連結'], ['No links', '無連結'],
    ['Successful Response', '成功回應'], ['Validation Error', '驗證錯誤'],
    ['Media type', '媒體類型'], ['Controls Accept header.', '設定 Accept 標頭。'],
    ['Example Value', '範例值'], ['Edit Value', '編輯內容'],
    ['Schema', '結構'], ['Schemas', '資料結構'], ['Expand all', '全部展開'],
    ['Collapse all', '全部收合'], ['Available values', '可選值'],
    ['Default value', '預設值'], ['Example', '範例'],
    ['Select the model displayed for this check. If omitted, the backend default is used. This endpoint does not run inference, change defaults, or affect model selection in POST /api/query.',
     '選擇本次檢查顯示的模型；未選擇時使用後端預設值。此端點不執行推論、不修改預設值，也不影響 POST /api/query 的模型選擇。'],
  ];
  const dictionary = new Map();
  for (const pair of pairs) {
    dictionary.set(pair[0], pair);
    dictionary.set(pair[1], pair);
    dictionary.set(pair[0] + ':', [pair[0] + ':', pair[1] + '：']);
    dictionary.set(pair[1] + '：', [pair[0] + ':', pair[1] + '：']);
  }
  let language = 'en';
  try { language = localStorage.getItem('wafer-docs-language') || 'en'; } catch {}
  if (!['zh', 'en'].includes(language)) language = 'en';
  select.value = language;
  const settings = {childList: true, subtree: true, characterData: true};
  const observer = new MutationObserver(translate);
  function translate() {
    observer.disconnect();
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    let node;
    while ((node = walker.nextNode())) {
      // Never alter payloads, field identifiers, editable inputs, or server output.
      if (node.parentElement.closest('pre, code, textarea, input, select, .response-col_status, .parameter__name, .model-title')) continue;
      const text = node.nodeValue;
      const pair = dictionary.get(text.trim());
      if (pair) {
        const value = pair[language === 'zh' ? 1 : 0];
        if (value !== text.trim()) node.nodeValue = text.replace(text.trim(), value);
      }
    }
    document.documentElement.lang = language === 'zh' ? 'zh-Hant' : 'en';
    document.title = language === 'zh' ? '晶圓查詢 API 文件' : 'Wafer React Query API - Swagger UI';
    observer.observe(root, settings);
  }
  select.addEventListener('change', () => {
    language = select.value;
    try { localStorage.setItem('wafer-docs-language', language); } catch {}
    translate();
  });
  translate();
})();
