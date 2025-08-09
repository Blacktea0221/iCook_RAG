# iCook_RAG
爬蟲練習   進行資料清整後向量化   使用RAG進行檢索   過程套用LLM model 提升使用體驗

使用ollama 內建的qwen3:4b-q4_K_M

執行:python -m scripts.app

前置條件:
1.連線進入dbeaver+postgresql

2.database資料夾 
透過postgresql.txt 在資料庫建立表格  
執行Text_conversion_vector.py  -> 食譜材料向量化
執行pgvector_bulk_upload.py -> 向量化資料上傳至postgres