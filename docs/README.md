# Documentation

The same content also lives in the [GitHub Wiki](https://github.com/jina-ai/jina-airgap/wiki). The wiki version renders the sidebar nav; this folder is the source of truth and what gets edited via pull requests.

## Pages

- [Home](Home.md) - overview and nav
- [Quick Start](Quick-Start.md) - 5-minute path with a prebuilt image
- [Bundling Guide](Bundling-Guide.md) - build your own, GCP L4 walkthrough
- [Model Catalog](Model-Catalog.md) - all 28 models, auto-generated from `models/catalog.json`
- [API Reference](API-Reference.md) - four schemas + multimodal + ES integration
- [Troubleshooting](Troubleshooting.md) - common errors and fixes

## Syncing to wiki

GitHub wikis must be initialized via the web UI (no API). Once initialized:

```bash
./scripts/sync-wiki.sh
```

That clones `jina-airgap.wiki.git`, copies these markdown files + `images/`, and pushes.

## Regenerating the Model Catalog

`Model-Catalog.md` is generated from [`models/catalog.json`](../models/catalog.json):

```bash
python3 scripts/gen_catalog_md.py > docs/Model-Catalog.md
```

Run this whenever you add a model or change a `prebuilt` entry in `scripts/gen_catalog_md.py`.
