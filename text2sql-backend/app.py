# app.py
import os, urllib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine
from langchain_openai import ChatOpenAI
from langchain_experimental.sql import SQLDatabaseChain
from langchain.sql_database import SQLDatabase
from fastapi.middleware.cors import CORSMiddleware
import re
from sqlalchemy import text
from fastapi import HTTPException
from langchain.chains import create_sql_query_chain

# app.py (最上方 imports 旁邊)
import re
from sqlalchemy import text


def strip_sql_code_fences(s: str) -> str:
    """
    去掉 ```sql ... ``` 或 ``` ... ``` 的 Markdown 程式碼區塊外殼
    """
    if not isinstance(s, str):
        return s
    s = s.strip()
    # 只移除最外層 code fence（含可選語言）
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_+-]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


# -------- 環境變數（從容器或 App Service 設定）--------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SQL_SERVER = os.getenv("SQL_SERVER")  # e.g. "text2sql-server.database.windows.net,1433"
SQL_DB = os.getenv("SQL_DB")  # e.g. "text2sql-db"
SQL_USER = os.getenv("SQL_USER")  # e.g. "lily3130@text2sql-server"
SQL_PASSWORD = os.getenv("SQL_PASSWORD")
ODBC_DRIVER = os.getenv("ODBC_DRIVER", "ODBC Driver 18 for SQL Server")

# -------- SQLAlchemy Engine --------
params = urllib.parse.quote_plus(
    f"DRIVER={{{ODBC_DRIVER}}};"
    f"SERVER={SQL_SERVER};DATABASE={SQL_DB};UID={SQL_USER};PWD={SQL_PASSWORD};"
    "Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;"
)
engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}")

# -------- LangChain：ChatOpenAI + SQLDatabaseChain --------
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

db = SQLDatabase(engine)
llm = ChatOpenAI(
    temperature=0, model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY
)

# 想看中間步驟/SQL就打開 return_intermediate_steps=True
# chain = SQLDatabaseChain.from_llm(llm, db, verbose=True, return_intermediate_steps=True)

# -------- FastAPI --------
app = FastAPI(title="Text-to-SQL API")

# 上線請把 * 改成你的前端網域
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://text2sqlfrontendstorage.z23.web.core.windows.net/",
    ],  # 開發期可先 *
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskIn(BaseModel):
    query: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(body: AskIn):
    try:
        question = (body.query or "").strip()
        if not question:
            raise HTTPException(status_code=400, detail="query must not be empty")

        # 只產生 SQL（不直接執行）
        sql_chain = create_sql_query_chain(llm, db)
        raw_sql = sql_chain.invoke({"question": question})
        sql = strip_sql_code_fences(raw_sql)

        # 簡單阻擋破壞性語句（可依需要再放寬/調整）
        lowered = sql.lower()
        if any(
            bad in lowered for bad in (" drop ", " delete ", " truncate ", " alter ")
        ):
            raise HTTPException(
                status_code=400,
                detail=f"Refusing to execute potentially destructive SQL: {sql}",
            )

        # 實際執行 SQL
        with engine.connect() as conn:
            result = conn.execute(text(sql))
            columns = list(result.keys())
            rows = [dict(r._mapping) for r in result]

        return {"ok": True, "sql": sql, "columns": columns, "rows": rows}

    except HTTPException:
        # 讓 FastAPI 正常回傳 HTTP 錯誤碼
        raise
    except Exception as e:
        # 其他錯誤用 ok=False 回前端
        return {"ok": False, "error": str(e)}


# test
