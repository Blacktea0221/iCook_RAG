# iCook_RAG
爬蟲練習   進行資料清整後向量化   使用RAG進行檢索   過程套用LLM model 提升使用體驗

使用ollama 內建的qwen3:4b-q4_K_M
未來使用openai api

python版本:3.10.11
建立虛擬環境:python -m venv venv
如果vscode有其他版本:py -3.10 -m venv venv
啟動虛擬環境:.\venv\Scripts\activate
退出虛擬環境:deactivate


列出已經安裝的所有套件和版本號:pip freeze > requirements.txt
下載套件指令:pip install -r requirements.txt 

CLI測試
CLI (RAG): python scripts/CLI_run/rag_cli.py
CLI (Agent): python scripts/CLI_run/agent_cli.py

目前特殊條件(素食葷食 不吃豬肉)  依賴本地端data/embeddings/Meat and Vegetarian.json   
資料沒有進資料庫  搜尋條件也沒有繼續做下去

API(調整中  要加入LangChain Agent)
uvicorn scripts.api.main:app --reload --port 8000
http://127.0.0.1:8000/docs


前置條件:
1.連線進入dbeaver+postgresql(本地端)

2.database資料夾 (如果沒有建立表格)
透過postgresql.txt 在資料庫建立表格  
執行Text_conversion_vector.py  -> 食譜材料向量化
執行pgvector_bulk_upload.py -> 向量化資料上傳至postgres


目標流程:
LINE Bot → POST /route(API)
   → API 裡呼叫 recipe_agent.build_agent()
      → LangChain Router (LLM) 判斷意圖(食譜,營養,價錢,辨識)  -> 目前意圖判斷寫死
         → 呼叫對應 Tool
            → RAG 查資料
            → 或查營養/價格（未來工具）
         → LLM 整理回覆
   → API 回傳給 LINE Bot
