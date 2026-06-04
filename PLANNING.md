# PLANNING - AI Model / POI Urban

## Vai tro cua thu muc nay

`poi_urban` phu trach model, data pipeline, nghien cuu va prototype AI. Day la noi phat trien cac thanh phan AI truoc khi dong goi sang backend.

Pham vi:

- Multimodal Encoder V4.
- Training/evaluation.
- Embedding generation.
- Semantic reranker.
- Demand proxy engine.
- Business location scoring prototype.
- Streamlit/app prototype neu can kiem thu nhanh.

## Nguyen tac nghien cuu

- Version 4 la baseline manh nhat ve semantic.
- Khong de geometry/spatial keo sai ngu nghia khi query co category ro.
- Moi metric phai gan voi use case san pham.
- Khong chi bao cao loss; can co retrieval/category purity/semantic mismatch.

## Module can co

```text
src/
  encoder/
    multimodal_encoder.py
  models/
    semantic_reranker.py
    demand_proxy_engine.py
    business_location_scorer.py
    itinerary_scoring.py
  data/
    category_normalizer.py
    poi_schema.py
  evaluation/
    retrieval_metrics.py
    business_metrics.py
```

## Huong cai tien model

### Semantic-first retrieval

Neu query co category ro nhu cafe, nha hang, an vat, quan nhau, text/category score phai duoc uu tien hon geometry.

### Hard negative mining

Cac POI gan nhau nhung khac intent/danh muc can bi xem la hard negative.

Vi du:

```text
Cafe vs quan nhau gan nhau ve vi tri -> khong nen gan nhau ve semantic embedding.
```

### Reranker

Sau cosine similarity, rerank bang:

```text
semantic_score
category_match
distance_score
rating_score
opening_score
route_score
competition/demand score neu role business
```

## Demand proxy engine

Khong tuyen bo mat do khach that. Tinh proxy:

```text
demand_proxy =
  review_count_signal
  + rating_quality
  + complementary_poi_density
  + review_keyword_signal
  + route_accessibility
  + in_app_behavior sau nay
```

## Business location scorer

Cong thuc baseline:

```text
opportunity_score =
  demand_proxy * 0.30
  + complementary_poi_score * 0.20
  + accessibility_score * 0.20
  + semantic_concept_fit * 0.20
  - competition_penalty * 0.10
```

Output phai co giai thich:

```json
{
  "area": "Hai Chau",
  "score": 0.81,
  "reason": "Nhieu POI bo tro, gan cum sinh vien/van phong, doi thu truc tiep vua phai.",
  "signals": {},
  "warnings": []
}
```

## Evaluation can co

- Recall@5/10.
- Category purity.
- Semantic mismatch rate.
- Itinerary route efficiency.
- Business opportunity sanity check.
- Ablation V1/V2/V3/V4.

## Dong goi sang backend

Backend khong nen phu thuoc vao notebook/prototype. Can export:

- Script precompute embeddings.
- File embeddings/cache.
- Module scoring co function ro.
- API/service wrapper neu can.

## Commit lien quan model

```text
feat(model): add demand proxy engine
feat(model): add semantic reranker
feat(model): add business location scorer
fix(model): prioritize category match for text queries
test(model): add scorer sanity cases
```

## Cau truc thu muc sau khi don dep

`	ext
main.py              Entrypoint train/evaluate chinh
app.py               Streamlit prototype
config.py            Cau hinh Python runtime
src/                 Model/data/training modules
research_pipeline/   Bao cao va visualization pipelinescripts/             Script phu tro, legacy scripts
logs/                Log runtime/local
results/             Output train/evaluation
` 

