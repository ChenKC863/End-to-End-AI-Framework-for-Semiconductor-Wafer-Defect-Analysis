# Local application operation / 本機操作

Terminal A manages Ollama, B runs FastAPI on port 8001, C runs Vite on port 5173, and D runs verification and builds. In development, Vite forwards /api requests to FastAPI. After building React and restarting FastAPI, port 8001 serves both the application and API. /docs is the API documentation, while / is the built React page.

OLLAMA_MODEL 是未提供 model 時的後端預設值，查詢送出的 model 優先。OLLAMA_NUM_GPU=0 讓本後端送出 CPU 模式選項，需在啟動 API 的終端機設定。/api/health 不執行模型推論，inference_checked: false 不代表模型故障或成功。

Source: Doc1.docx terminal setup; wafer_llm_query/react_api.py; frontend/vite.config.js.
