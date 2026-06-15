import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "analise_filas.db")


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def criar_tabela():
    conn = get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS historico_filas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            quantidade INTEGER NOT NULL,
            tempo_estimado INTEGER NOT NULL,
            entradas INTEGER DEFAULT 0,
            saidas INTEGER DEFAULT 0,
            tempo_real_medio REAL DEFAULT NULL,
            created_at TEXT DEFAULT (datetime('now','localtime'))
        )
    """)
    conn.commit()
    conn.close()


def salvar_metrica(quantidade, tempo_estimado, entradas, saidas, tempo_real_medio=None):
    conn = get_conn()
    conn.execute(
        "INSERT INTO historico_filas (quantidade, tempo_estimado, entradas, saidas, tempo_real_medio) "
        "VALUES (?, ?, ?, ?, ?)",
        (quantidade, tempo_estimado, entradas, saidas, tempo_real_medio),
    )
    conn.commit()
    conn.close()


def listar_historico(pagina=1, por_pagina=25):
    conn = get_conn()
    offset = (pagina - 1) * por_pagina
    rows = conn.execute(
        "SELECT * FROM historico_filas ORDER BY created_at DESC LIMIT ? OFFSET ?",
        (por_pagina, offset),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def contar_historico():
    conn = get_conn()
    total = conn.execute("SELECT COUNT(*) AS total FROM historico_filas").fetchone()["total"]
    conn.close()
    return total


def estatisticas():
    conn = get_conn()
    stats = dict(conn.execute("""
        SELECT
            COUNT(*) AS total_registros,
            ROUND(AVG(quantidade), 1) AS media_pessoas,
            MAX(quantidade) AS max_pessoas,
            MIN(quantidade) AS min_pessoas,
            ROUND(AVG(tempo_estimado), 1) AS media_tempo_est,
            MAX(tempo_estimado) AS max_tempo_est,
            ROUND(AVG(tempo_real_medio), 1) AS media_tempo_real,
            MAX(tempo_real_medio) AS max_tempo_real,
            SUM(entradas) AS total_entradas,
            SUM(saidas) AS total_saidas
        FROM historico_filas
    """).fetchone())

    crescimento = [
        dict(r) for r in conn.execute(
            "SELECT quantidade, tempo_real_medio, created_at FROM historico_filas ORDER BY created_at ASC"
        ).fetchall()
    ]
    conn.close()

    if crescimento:
        stats["crescimento"] = crescimento[-1]["quantidade"] - crescimento[0]["quantidade"]
        stats["dados_crescimento"] = crescimento
    else:
        stats["crescimento"] = 0
        stats["dados_crescimento"] = []

    return stats
