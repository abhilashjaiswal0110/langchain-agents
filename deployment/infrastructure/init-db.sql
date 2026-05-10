-- PostgreSQL initialisation for LangChain Enterprise Agents Platform
-- Applied once when the container is first created.

-- Sessions table (mirrors SQLiteSessionStore schema)
CREATE TABLE IF NOT EXISTS sessions (
    session_id  TEXT        NOT NULL,
    tenant_id   TEXT        NOT NULL DEFAULT 'default',
    data        TEXT        NOT NULL,
    expires_at  TIMESTAMPTZ,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (session_id, tenant_id)
);

CREATE INDEX IF NOT EXISTS idx_sessions_tenant ON sessions (tenant_id);
CREATE INDEX IF NOT EXISTS idx_sessions_expires ON sessions (expires_at)
    WHERE expires_at IS NOT NULL;

-- Conversation messages table (append-only audit log)
CREATE TABLE IF NOT EXISTS messages (
    id          BIGSERIAL   PRIMARY KEY,
    session_id  TEXT        NOT NULL,
    tenant_id   TEXT        NOT NULL DEFAULT 'default',
    role        TEXT        NOT NULL CHECK (role IN ('user', 'assistant', 'system', 'tool')),
    content     TEXT        NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages (session_id, tenant_id, created_at);

-- Cost tracking table
CREATE TABLE IF NOT EXISTS token_usage (
    id              BIGSERIAL   PRIMARY KEY,
    session_id      TEXT,
    tenant_id       TEXT        NOT NULL DEFAULT 'default',
    agent_type      TEXT        NOT NULL,
    model           TEXT        NOT NULL,
    input_tokens    INTEGER     NOT NULL DEFAULT 0,
    output_tokens   INTEGER     NOT NULL DEFAULT 0,
    cost_usd        NUMERIC(12, 8) NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_usage_tenant_agent ON token_usage (tenant_id, agent_type, created_at);
