# Recent Changes

## Version 2.0 - Modular Architecture (June 2026)

### Phase 2.1: Data Abstraction & Configuration System
**Commit:** "Add modular ingestion pipeline and unified configuration system"

**Added:**
- Abstract data layer with pluggable connectors (CSV, JSONL, Directory)
- Modular agent system with base classes and factory pattern
- YAML-based configuration for complete pipeline
- Universal ingestion supporting any data source/domain
- Comprehensive documentation (INGESTION_GUIDE, PIPELINE_GUIDE, ARCHITECTURE)
- New unified entry point: `run_pipeline.py`
- Test suite: `test_ingestion_abstraction.py`

**Key Files:**
- `src/ingestion/` - Abstract data connectors
- `src/agents/base.py` - Agent base classes
- `src/agents/factory.py` - Agent factory pattern
- `src/config/pipeline_config.py` - Configuration system
- `src/pipeline/orchestrator.py` - Workflow orchestration
- `configs/*.yaml` - Example configurations

### Phase 2.2: Integration & Cleanup
**Commit:** "Integrate API with new architecture and cleanup"

**Changed:**
- ✅ `api/main.py` - Now uses `PipelineOrchestrator` instead of direct agent imports
- ✅ `hf_space/app.py` - Updated to use new pipeline system
- ✅ README.md - Replaced with modern, framework-focused documentation
- ✅ .gitignore - Added archive/ directory

**Removed:**
- ❌ `src/agents/critic.py` - Archived (superseded by critic_v3)
- ❌ `src/agents/critic_v2.py` - Archived (superseded by critic_v3)
- ❌ Old README - Moved to archive/README_OLD.md
- ❌ ANALYSIS.md - Moved to archive/

**Added:**
- ✅ API_GUIDE.md - Complete API documentation
- ✅ Basic API authentication (optional API key)
- ✅ Better error handling and validation
- ✅ Configuration-driven API

**Impact:**
- API and Web UI now use the new modular architecture
- Consistent configuration across all entry points (CLI, API, Web)
- Cleaner codebase with archived legacy files
- Better documentation for users and developers

---

## Migration Guide (For Existing Users)

### If you used `main.py`:
**Old:**
```bash
python main.py
```

**New:**
```bash
python run_pipeline.py --config configs/pipeline_query_only.yaml --interactive
```

### If you used `ingest.py`:
**Old:**
```bash
python ingest.py --data-dir ./data/amazon_data
```

**New:**
```bash
python run_pipeline.py --config configs/pipeline_amazon_reviews.yaml
```

Or use the new flexible ingestion:
```bash
python ingest_universal.py --config configs/amazon_reviews.yaml
```

### If you used the API:
**Old:**
```python
# API loaded agents directly from src/agents/
```

**New:**
```python
# API now uses PipelineOrchestrator with YAML config
# Set TGA_CONFIG environment variable to choose config
export TGA_CONFIG=configs/pipeline_query_only.yaml
uvicorn api.main:app
```

### Configuration Changes:
**Old:** Settings in `config/settings.py` via environment variables

**New:** YAML configuration files in `configs/`
- Complete pipeline control
- Multiple configs for different scenarios
- Version-controllable
- Human-readable

---

## Breaking Changes

### Removed:
- Direct imports of `SimplifiedCriticAgent` (use factory: `create_critic()`)
- Direct imports of old `CriticAgent` (archived)
- Hardcoded brand lists (now in YAML configs)

### Changed:
- Agent constructors now accept `config` dict parameter
- API endpoint responses have slightly different structure
- Configuration must be provided via YAML (not just env vars)

### Deprecated (Still Works):
- `main.py` - Still functional but use `run_pipeline.py` instead
- `ingest.py` - Still functional but use `ingest_universal.py` instead
- `config/settings.py` - Still used internally but prefer YAML configs

---

## What's Next?

### Recommended (Before Public Release):
1. Add rate limiting to API
2. Add async support for better concurrency
3. Create Docker image
4. Add integration tests
5. Set up CI/CD pipeline

### Future Enhancements:
1. More data connectors (PostgreSQL, MongoDB, Twitter API)
2. More LLM backends (OpenAI, Anthropic, Cohere)
3. GraphQL API
4. Real-time streaming ingestion
5. Advanced visualizations
6. Multi-tenancy support

---

## Documentation

- **README.md** - Project overview and quick start
- **ARCHITECTURE.md** - System design and extension points
- **INGESTION_GUIDE.md** - Data ingestion details
- **PIPELINE_GUIDE.md** - End-to-end workflow guide
- **API_GUIDE.md** - REST API documentation

---

**Questions?** Open an issue on GitHub or check the documentation.
