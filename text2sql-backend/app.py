# app.py
import os, re, io, urllib
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_experimental.sql import SQLDatabaseChain
from langchain.sql_database import SQLDatabase
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text, create_engine
from sqlalchemy.engine import Engine
from fastapi import HTTPException
from langchain.chains import create_sql_query_chain
from typing import Optional, Literal, List, Dict, Any
import pandas as pd


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


def relax_numeric_casts(sql: str) -> str:
    """
    將容易出錯的 CAST/CONVERT 改成 TRY_CONVERT / TRY_CAST，
    以避免 'Error converting data type varchar to float.' 之類錯誤。
    """
    if not isinstance(sql, str):
        return sql
    s = sql

    # CAST(x AS FLOAT/DECIMAL/INT/NUMERIC) -> TRY_CAST(x AS ...)
    s = re.sub(r"\bCAST\s*\(", "TRY_CAST(", s, flags=re.I)

    # CONVERT(FLOAT/DECIMAL/INT/NUMERIC, x) -> TRY_CONVERT(...)
    s = re.sub(r"\bCONVERT\s*\(", "TRY_CONVERT(", s, flags=re.I)

    # 也順手把 double precision 一律視為 FLOAT
    s = re.sub(r"\bDOUBLE\s+PRECISION\b", "FLOAT", s, flags=re.I)

    return s


def normalize_sql_for_mssql(sql: str) -> str:
    """
    - LIMIT n OFFSET m   -> OFFSET m ROWS FETCH NEXT n ROWS ONLY (需已有 ORDER BY，否則退回 TOP n)
    - LIMIT m, n         -> OFFSET m ROWS FETCH NEXT n ROWS ONLY (同上)
    - LIMIT n            -> 在 SELECT 後注入 TOP n
    - `identifier`       -> [identifier]
    """
    if not isinstance(sql, str):
        return sql
    s = sql.strip().rstrip(";")

    def has_order_by(x: str) -> bool:
        return re.search(r"\border\s+by\b", x, flags=re.I) is not None

    # 1) LIMIT n OFFSET m
    m = re.search(r"\blimit\s+(\d+)\s+offset\s+(\d+)\b", s, flags=re.I)
    if m:
        n = int(m.group(1))
        off = int(m.group(2))
        if has_order_by(s):
            s = re.sub(
                r"\blimit\s+\d+\s+offset\s+\d+\b",
                f"OFFSET {off} ROWS FETCH NEXT {n} ROWS ONLY",
                s,
                flags=re.I,
            )
        else:
            # 沒有 ORDER BY 時，OFFSET 不合法：退回 TOP n
            s = re.sub(r"\blimit\s+\d+\s+offset\s+\d+\b", "", s, flags=re.I)
            s = re.sub(r"^\s*select\s+", f"SELECT TOP {n} ", s, flags=re.I)
        # 反引號 -> 中括號
        s = re.sub(r"`([^`]+)`", r"[\1]", s)
        return s

    # 2) LIMIT m, n
    m = re.search(r"\blimit\s+(\d+)\s*,\s*(\d+)\b", s, flags=re.I)
    if m:
        off = int(m.group(1))
        n = int(m.group(2))
        if has_order_by(s):
            s = re.sub(
                r"\blimit\s+\d+\s*,\s*\d+\b",
                f"OFFSET {off} ROWS FETCH NEXT {n} ROWS ONLY",
                s,
                flags=re.I,
            )
        else:
            s = re.sub(r"\blimit\s+\d+\s*,\s*\d+\b", "", s, flags=re.I)
            s = re.sub(r"^\s*select\s+", f"SELECT TOP {n} ", s, flags=re.I)
        s = re.sub(r"`([^`]+)`", r"[\1]", s)
        return s

    # 3) LIMIT n（常見）
    m = re.search(r"\blimit\s+(\d+)\b", s, flags=re.I)
    if m:
        n = int(m.group(1))
        s = re.sub(r"\blimit\s+\d+\b", "", s, flags=re.I)
        s = re.sub(r"^\s*select\s+", f"SELECT TOP {n} ", s, flags=re.I)
        s = re.sub(r"`([^`]+)`", r"[\1]", s)
        return s

    # 4) 僅處理反引號
    s = re.sub(r"`([^`]+)`", r"[\1]", s)
    return s


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
engine = create_engine(
    f"mssql+pyodbc:///?odbc_connect={params}",
    fast_executemany=True,
)

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
        "https://text2sqlfrontendstorage.z23.web.core.windows.net",
        # "*"
    ],  # 開發期可先 *
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskIn(BaseModel):
    query: str
    use_enrichment: bool = False  # 前端切換 Text Enrichment
    table_whitelist: Optional[List[str]] = None  # 可選：限制可用表


# ------------------- Schema/Glossary helpers -------------------
def get_schema_snapshot(conn) -> Dict[str, List[str]]:
    """
    讀取 INFORMATION_SCHEMA，回傳 {table: [columns...]}
    """
    q = """
    SELECT TABLE_NAME, COLUMN_NAME
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_CATALOG = :db
    ORDER BY TABLE_NAME, ORDINAL_POSITION
    """
    rows = conn.execute(text(q), {"db": SQL_DB}).fetchall()
    schema: Dict[str, List[str]] = {}
    for t, c in rows:
        schema.setdefault(t, []).append(c)
    return schema


def load_business_glossary(conn) -> List[Dict[str, Any]]:
    """
    嘗試讀取 BUSINESS_GLOSSARY(term, synonyms, columns)（可選）
    synonyms/columns 建議用逗號分隔字串或 JSON；這裡通用處理
    """
    try:
        rows = conn.execute(
            text("SELECT term, synonyms, columns FROM BUSINESS_GLOSSARY")
        ).fetchall()
    except Exception:
        return []
    items = []
    for r in rows:
        term = r[0]
        synonyms = r[1]
        cols = r[2]

        def to_list(v):
            if v is None:
                return []
            s = str(v).strip()
            if s.startswith("[") and s.endswith("]"):
                # 粗略處理 JSON 風；不嚴格解析避免引入 json 依賴
                s = s.strip("[]")
                parts = [x.strip().strip("'\"") for x in s.split(",") if x.strip()]
                return parts
            return [x.strip() for x in s.split(",") if x.strip()]

        items.append(
            {
                "term": term,
                "synonyms": to_list(synonyms),
                "columns": to_list(cols),
            }
        )
    return items


def build_enrichment_prompt(
    question: str,
    schema: Dict[str, List[str]],
    glossary: List[Dict[str, Any]],
    table_whitelist: Optional[List[str]],
) -> str:
    """
    產生一段系統化的 prompt，請 LLM 回傳 enriched user question 與映射
    """
    # 過濾 schema（若有白名單）
    if table_whitelist:
        schema = {t: cols for t, cols in schema.items() if t in set(table_whitelist)}

    schema_lines = []
    for t, cols in schema.items():
        schema_lines.append(
            f"- {t}({', '.join(cols[:60])}{'...' if len(cols)>60 else ''})"
        )

    glossary_lines = []
    for item in glossary:
        glossary_lines.append(
            f"- {item['term']} -> synonyms: {', '.join(item['synonyms']) or '(none)'}; columns: {', '.join(item['columns']) or '(unspecified)'}"
        )

    prompt = f"""
You are a SQL text enrichment assistant.
Task:
1) Map ambiguous business terms to concrete table.column where possible.
2) Expand the user question with synonyms and explicit filters (units, date grain).
3) Keep *facts* untouched; do not invent tables/columns that don't exist in schema.
4) Return STRICT JSON with keys: enriched_question, term_mappings (list of {{term, mapped_to}}), assumptions (list), filters (list).

User question:
{question}

Schema (tables and columns):
{os.linesep.join(schema_lines) if schema_lines else '(empty)'}

Business glossary (optional):
{os.linesep.join(glossary_lines) if glossary_lines else '(none)'}
"""
    return prompt


def enrich_question_with_llm(
    question: str, engine: Engine, table_whitelist: Optional[List[str]]
) -> Dict[str, Any]:
    """
    回傳 { enriched_question: str, debug: {...} }
    """
    with engine.connect() as conn:
        schema = get_schema_snapshot(conn)
        glossary = load_business_glossary(conn)

    prompt = build_enrichment_prompt(question, schema, glossary, table_whitelist)
    # 讓回傳更穩定：要求 JSON
    sys_msg = "Answer in JSON only. No extra commentary."
    resp = llm.invoke(
        [{"role": "system", "content": sys_msg}, {"role": "user", "content": prompt}]
    )

    content = (resp.content or "").strip()

    # 嘗試解析 JSON；若失敗則退回原始問題（不會阻塞整體流程）
    import json

    enriched_question = question
    debug: Dict[str, Any] = {}
    try:
        data = json.loads(content)
        if isinstance(data, dict) and "enriched_question" in data:
            enriched_question = data["enriched_question"] or question
            debug = {
                "term_mappings": data.get("term_mappings", []),
                "assumptions": data.get("assumptions", []),
                "filters": data.get("filters", []),
                "raw": data,
            }
    except Exception:
        debug = {"parse_error": True, "raw": content}

    return {"enriched_question": enriched_question, "debug": debug}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(body: AskIn):
    try:
        question = (body.query or "").strip()
        if not question:
            raise HTTPException(status_code=400, detail="query must not be empty")

        # 1) 可選的 Text Enrichment
        debug_enrich = None
        if body.use_enrichment:
            enriched = enrich_question_with_llm(question, engine, body.table_whitelist)
            question = enriched["enriched_question"] or question
            debug_enrich = enriched["debug"]

        # 2) 產生 SQL（不直接執行）
        sql_chain = create_sql_query_chain(llm, db)
        raw_sql = sql_chain.invoke({"question": question})
        sql = strip_sql_code_fences(raw_sql)

        # 把 LIMIT/反引號 轉成 SQL Server 可用語法
        sql = normalize_sql_for_mssql(sql)
        # <<< 新增：把 CAST/CONVERT 換成 TRY_*，避免 varchar -> float 轉型爆炸
        sql = relax_numeric_casts(sql)

        # 3) 簡單阻擋破壞性語句
        lowered = f" {sql.lower()} "
        if any(
            bad in lowered
            for bad in (" drop ", " delete ", " truncate ", " alter ", " update ")
        ):
            raise HTTPException(
                status_code=400,
                detail=f"Refusing to execute potentially destructive SQL: {sql}",
            )

        # 4) 執行查詢
        with engine.connect() as conn:
            result = conn.execute(text(sql))
            columns = list(result.keys())
            rows = [dict(r._mapping) for r in result]

        return {
            "ok": True,
            "sql": sql,
            "columns": columns,
            "rows": rows,
            "enrichment": debug_enrich if body.use_enrichment else None,
        }

    except HTTPException:
        raise
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ------------------- File Upload → Azure SQL -------------------
# 需求：
#   pip 安裝 python-multipart、pandas、openpyxl（處理 xlsx）
# 參數：
#   file: CSV 或 Excel
#   table_name: 可選；若未提供，預設用檔名
#   if_exists: 'fail' | 'replace' | 'append'（預設 replace）
#   sheet_name: 讀 Excel 時可選（不填則全部 sheet）
def sanitize_identifier(name: str) -> str:
    name = name.strip()
    name = re.sub(r"[^\w]+", "_", name)
    name = re.sub(r"(^_+|_+$)", "", name)
    if not name:
        name = "uploaded_table"
    return name


@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    table_name: Optional[str] = Form(None),
    sheet_name: Optional[str] = Form(None),
):
    """
    上傳 CSV / Excel 並寫入 Azure SQL
    - 一律覆蓋 (replace)
    - 欄名清理、year 正規化、數值智慧轉型
    - 回傳：匯入的表名清單 + 表數 + 總列數
    """
    try:
        filename = (file.filename or "upload").strip()
        ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
        content = await file.read()
        bio = io.BytesIO(content)

        # 1) 讀檔 → dict[str, DataFrame]
        dfs: Dict[str, pd.DataFrame] = {}

        if ext in ("csv", "txt"):
            # 先以字串讀，避免一開始就轉成數值造成資訊遺失
            df = pd.read_csv(bio, dtype=str, keep_default_na=False, na_values=[""])
            # 用檔名（去副檔名）當表名，或是外部傳入的 table_name
            tname = table_name or sanitize_identifier(filename.rsplit(".", 1)[0])
            dfs[tname] = df

        elif ext in ("xlsx", "xls"):
            xls = pd.ExcelFile(bio)
            targets = xls.sheet_names if not sheet_name else [sheet_name]
            for sh in targets:
                if "content" in sh.lower():
                    continue
                df = pd.read_excel(xls, sheet_name=sh, dtype=str)
                dfs[sanitize_identifier(sh)] = df

            # 若強制指定了 table_name，且只有一張表 → 用這個名字覆蓋
            if table_name and len(dfs) == 1:
                only_key = next(iter(dfs.keys()))
                dfs = {sanitize_identifier(table_name): dfs[only_key]}
        else:
            raise HTTPException(
                status_code=400, detail="Only CSV/XLSX/XLS are supported"
            )

        if not dfs:
            raise HTTPException(
                status_code=400, detail="No worksheet or data found to import"
            )

        # 2) 欄名/資料清理
        def smart_cast_numeric(df: pd.DataFrame) -> pd.DataFrame:
            for c in df.columns:
                s = df[c].astype(str).str.strip()
                s = s.replace(
                    {
                        "-": None,
                        "—": None,
                        "–": None,
                        "N.A.": None,
                        "n.a.": None,
                        "NA": None,
                        "na": None,
                        "": None,
                    }
                )
                s = s.str.replace(",", "", regex=False)  # 去千分位
                num = pd.to_numeric(s, errors="coerce")
                # > 60% 能轉才採用數值型，否則保留原字串
                if num.notna().mean() >= 0.6:
                    df[c] = num
                else:
                    df[c] = s.fillna("")  # 維持乾淨字串
            return df

        with engine.begin() as conn:
            for tname, df in dfs.items():
                t_final = (
                    sanitize_identifier(table_name)
                    if table_name and len(dfs) == 1
                    else sanitize_identifier(tname)
                )

                # 欄名清理
                df.columns = (
                    df.columns.astype(str)
                    .str.strip()
                    .str.lower()
                    .str.replace(r"[^\w]+", "_", regex=True)
                    .str.replace(r"(^_+|_+$)", "", regex=True)
                )

                # year 欄位正規化
                CANON = {"eimh_year": "year", "yr": "year"}
                rename_map = {}
                for c in list(df.columns):
                    key = c.lower()
                    if key in CANON:
                        rename_map[c] = CANON[key]
                    elif re.search(r"(?:^|_)year(?:_|$)", key):
                        rename_map[c] = "year"
                df.rename(columns=rename_map, inplace=True)

                # 丟掉全空列 / 全空欄
                df.dropna(how="all", inplace=True)
                df.dropna(axis=1, how="all", inplace=True)

                # 數值智慧轉型
                df = smart_cast_numeric(df)

                # 3) 先 DROP（保險），再 replace
                safe_name = f"dbo.{t_final}"
                conn.execute(
                    text(f"IF OBJECT_ID(:obj, 'U') IS NOT NULL DROP TABLE {safe_name}"),
                    {"obj": safe_name},
                )

                df.to_sql(
                    t_final,
                    conn,
                    if_exists="replace",  # 永遠覆蓋
                    index=False,
                    chunksize=1000,
                    schema="dbo",
                )

        # 3) 回傳統計
        total_tables = len(dfs)
        total_rows = int(sum(len(df) for df in dfs.values()))
        return {
            "ok": True,
            "message": f"Imported {total_tables} table(s), {total_rows} row(s).",
            "tables": (
                list(dfs.keys())
                if not (table_name and len(dfs) == 1)
                else [sanitize_identifier(table_name)]
            ),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
