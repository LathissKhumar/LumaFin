-- LumaFin schema (aligned to current code usage)

CREATE TABLE IF NOT EXISTS users (
    user_id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE,
    created_at TIMESTAMP DEFAULT NOW(),
    settings JSONB DEFAULT '{}'::jsonb
);

-- Transactions used by AMPT and feedback
CREATE TABLE IF NOT EXISTS transactions (
    id SERIAL PRIMARY KEY,
    user_id INT REFERENCES users(user_id),
    merchant TEXT NOT NULL,
    amount NUMERIC(12,2),
    description TEXT,
    date TIMESTAMP,
    embedding vector(384),
    predicted_category VARCHAR(100),
    confidence FLOAT,
    is_corrected BOOLEAN DEFAULT FALSE,
    correct_category VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_user_transactions ON transactions(user_id, date DESC);
CREATE INDEX IF NOT EXISTS idx_txn_embedding ON transactions USING ivfflat (embedding vector_cosine_ops);

-- Personal centroids for AMPT
CREATE TABLE IF NOT EXISTS personal_centroids (
    id SERIAL PRIMARY KEY,
    user_id INT REFERENCES users(user_id),
    category_name VARCHAR(100) NOT NULL,
    centroid_vector JSONB, -- Use JSONB list[float] here; swap to vector(384) if pgvector desired
    num_transactions INT DEFAULT 1,
    metadata JSONB,
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(user_id, category_name)
);

CREATE INDEX IF NOT EXISTS idx_user_centroids ON personal_centroids(user_id);

-- Global taxonomy
CREATE TABLE IF NOT EXISTS global_taxonomy (
    category_id SERIAL PRIMARY KEY,
    category_name VARCHAR(100) UNIQUE NOT NULL,
    parent_category_id INT REFERENCES global_taxonomy(category_id),
    level INT,
    description TEXT,
    example_keywords TEXT[]
);

-- Global labeled examples
CREATE TABLE IF NOT EXISTS global_examples (
    example_id SERIAL PRIMARY KEY,
    category_id INT REFERENCES global_taxonomy(category_id),
    text TEXT NOT NULL,
    embedding vector(384),
    source VARCHAR(50) DEFAULT 'seed',
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_category_examples ON global_examples(category_id);
CREATE INDEX IF NOT EXISTS idx_example_embedding ON global_examples USING ivfflat (embedding vector_cosine_ops);

-- Rule engine patterns
CREATE TABLE IF NOT EXISTS rules (
    rule_id SERIAL PRIMARY KEY,
    pattern TEXT NOT NULL,
    category_name VARCHAR(100) NOT NULL,
    priority INT DEFAULT 1,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Feedback queue for continuous learning
CREATE TABLE IF NOT EXISTS feedback_queue (
    feedback_id SERIAL PRIMARY KEY,
    user_id INT REFERENCES users(user_id),
    transaction_id INT REFERENCES transactions(id),
    predicted_category VARCHAR(100),
    correct_category VARCHAR(100),
    processed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_unprocessed_feedback ON feedback_queue(processed, created_at);
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Transactions table with vector embeddings
CREATE TABLE IF NOT EXISTS transactions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    merchant TEXT NOT NULL,
    amount DECIMAL(10, 2) NOT NULL,
    date DATE NOT NULL,
    description TEXT,
    category VARCHAR(100),
    embedding vector(384),  -- all-MiniLM-L6-v2 produces 384-dim vectors
    confidence FLOAT,
    is_personal_category BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_user_transactions ON transactions(user_id, date DESC);
CREATE INDEX idx_embedding ON transactions USING ivfflat (embedding vector_cosine_ops);

-- Personal centroids for AMPT clustering
CREATE TABLE IF NOT EXISTS personal_centroids (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    category_name VARCHAR(100) NOT NULL,
    centroid_vector vector(384) NOT NULL,
    metadata JSONB,  -- {merchant_pattern, time_pattern, amount_range, num_transactions}
    quality_score FLOAT,  -- silhouette score
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, category_name)
);

CREATE INDEX idx_user_centroids ON personal_centroids(user_id);
CREATE INDEX idx_centroid_vector ON personal_centroids USING ivfflat (centroid_vector vector_cosine_ops);

-- Global taxonomy
CREATE TABLE IF NOT EXISTS global_taxonomy (
    id SERIAL PRIMARY KEY,
    category_name VARCHAR(100) UNIQUE NOT NULL,
    parent_category VARCHAR(100),
    description TEXT,
    level INTEGER DEFAULT 1
);

-- Global labeled examples
CREATE TABLE IF NOT EXISTS global_examples (
    id SERIAL PRIMARY KEY,
    merchant TEXT NOT NULL,
    amount DECIMAL(10, 2),
    description TEXT,
    category_id INTEGER REFERENCES global_taxonomy(id),
    embedding vector(384),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_category_examples ON global_examples(category_id);
CREATE INDEX idx_example_embedding ON global_examples USING ivfflat (embedding vector_cosine_ops);

-- Rule engine patterns
CREATE TABLE IF NOT EXISTS rules (
    id SERIAL PRIMARY KEY,
    pattern TEXT NOT NULL,
    category_name VARCHAR(100) NOT NULL,
    priority INTEGER DEFAULT 0,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Feedback queue for continuous learning
CREATE TABLE IF NOT EXISTS feedback_queue (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    transaction_id INTEGER REFERENCES transactions(id),
    old_category VARCHAR(100),
    new_category VARCHAR(100) NOT NULL,
    processed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_unprocessed_feedback ON feedback_queue(processed, created_at);

-- Insert some default categories
INSERT INTO global_taxonomy (category_name, parent_category, description, level) VALUES
('Food & Dining', NULL, 'Restaurants, groceries, cafes', 1),
('Transportation', NULL, 'Gas, parking, public transit', 1),
('Shopping', NULL, 'Retail purchases', 1),
('Entertainment', NULL, 'Movies, games, hobbies', 1),
('Bills & Utilities', NULL, 'Rent, electricity, internet', 1),
('Healthcare', NULL, 'Medical, pharmacy, insurance', 1),
('Travel', NULL, 'Hotels, flights, vacation', 1),
('Income', NULL, 'Salary, refunds, transfers in', 1),
('Uncategorized', NULL, 'Unknown or unclassified', 1)
ON CONFLICT (category_name) DO NOTHING;

-- Insert some example rules
INSERT INTO rules (pattern, category_name, priority) VALUES
('(?i)netflix|hulu|spotify|disney\+', 'Entertainment', 100),
('(?i)whole foods|kroger|safeway|trader joe', 'Food & Dining', 100),
('(?i)shell|chevron|exxon|bp gas', 'Transportation', 100),
('(?i)atm withdrawal|atm fee', 'Cash', 90),
('(?i)amazon|amazon\.com|amzn', 'Shopping', 80)
ON CONFLICT DO NOTHING;
