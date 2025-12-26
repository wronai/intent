-- IntentForge Database Initialization
-- Creates core tables for intent caching and code storage

-- =============================================================================
-- Extensions
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For text search

-- =============================================================================
-- Intent Cache Table
-- =============================================================================

CREATE TABLE IF NOT EXISTS intent_cache (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    fingerprint VARCHAR(32) UNIQUE NOT NULL,
    description TEXT NOT NULL,
    intent_type VARCHAR(50) NOT NULL,
    target_platform VARCHAR(50) NOT NULL,
    context JSONB DEFAULT '{}',
    generated_code TEXT,
    language VARCHAR(20),
    validation_passed BOOLEAN DEFAULT FALSE,
    hit_count INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_intent_cache_fingerprint ON intent_cache(fingerprint);
CREATE INDEX idx_intent_cache_created_at ON intent_cache(created_at);
CREATE INDEX idx_intent_cache_intent_type ON intent_cache(intent_type);

-- =============================================================================
-- Generated Code History
-- =============================================================================

CREATE TABLE IF NOT EXISTS code_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    intent_id UUID REFERENCES intent_cache(id),
    code TEXT NOT NULL,
    language VARCHAR(20) NOT NULL,
    version INTEGER DEFAULT 1,
    validation_result JSONB,
    execution_result JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_code_history_intent_id ON code_history(intent_id);

-- =============================================================================
-- Schema Registry
-- =============================================================================

CREATE TABLE IF NOT EXISTS schema_registry (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) UNIQUE NOT NULL,
    schema_type VARCHAR(50) NOT NULL,
    schema_definition JSONB NOT NULL,
    version INTEGER DEFAULT 1,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================================
-- Usage Analytics
-- =============================================================================

CREATE TABLE IF NOT EXISTS usage_analytics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    client_id VARCHAR(100),
    intent_type VARCHAR(50),
    target_platform VARCHAR(50),
    processing_time_ms FLOAT,
    cache_hit BOOLEAN DEFAULT FALSE,
    validation_passed BOOLEAN,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_usage_analytics_created_at ON usage_analytics(created_at);
CREATE INDEX idx_usage_analytics_client_id ON usage_analytics(client_id);

-- =============================================================================
-- Triggers
-- =============================================================================

-- Auto-update updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_intent_cache_updated_at
    BEFORE UPDATE ON intent_cache
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_schema_registry_updated_at
    BEFORE UPDATE ON schema_registry
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Increment hit count on cache access
CREATE OR REPLACE FUNCTION increment_cache_hit()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE intent_cache 
    SET hit_count = hit_count + 1 
    WHERE fingerprint = NEW.fingerprint;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- =============================================================================
-- Sample Data (for testing)
-- =============================================================================

-- Insert some default schemas
INSERT INTO schema_registry (name, schema_type, schema_definition) VALUES
('intent_request', 'core', '{"type": "object", "required": ["description"]}'),
('form_data', 'form', '{"type": "object", "required": ["form_id", "fields"]}'),
('sql_query', 'database', '{"type": "object", "required": ["operation"]}')
ON CONFLICT (name) DO NOTHING;

-- =============================================================================
-- Views
-- =============================================================================

-- Cache statistics
CREATE OR REPLACE VIEW cache_stats AS
SELECT 
    intent_type,
    target_platform,
    COUNT(*) as total_cached,
    SUM(hit_count) as total_hits,
    AVG(hit_count) as avg_hits,
    MIN(created_at) as first_cached,
    MAX(updated_at) as last_accessed
FROM intent_cache
GROUP BY intent_type, target_platform;

-- Daily usage
CREATE OR REPLACE VIEW daily_usage AS
SELECT 
    DATE(created_at) as date,
    intent_type,
    COUNT(*) as requests,
    SUM(CASE WHEN cache_hit THEN 1 ELSE 0 END) as cache_hits,
    AVG(processing_time_ms) as avg_time_ms
FROM usage_analytics
GROUP BY DATE(created_at), intent_type
ORDER BY date DESC;

-- =============================================================================
-- Grants (if using separate app user)
-- =============================================================================

-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO intentforge_app;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO intentforge_app;
