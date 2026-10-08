Elasticsearch calls jina-on-prem through its [inference API](https://www.elastic.co/docs/api/doc/elasticsearch/group/endpoint-inference). You register the container once as an inference endpoint, and from then on Elasticsearch sends it text at index time and at query time. Your application only talks to Elasticsearch.

jina-on-prem speaks the protocols of two services Elasticsearch already has built in:

| Model type | Elasticsearch `service` | Task type | Path on the container |
|---|---|---|---|
| Embeddings | `openai` | `text_embedding` | `/v1/embeddings` |
| Rerankers | `cohere` | `rerank` | `/v1/rerank` or `/v2/rerank` |

Every request on this page was run end to end against Elasticsearch 9.5.3 and 9.0.8, with `jina-embeddings-v5-text-nano` and `jina-reranker-v3.5` on CPU.

## Before you start

- **Elasticsearch 9.0 or later.** `semantic_text` is generally available from 9.0.
- **A license that includes the inference API.** On the free Basic license, registering an endpoint fails with `current license is non-compliant for [inference]`, and a reranking query fails the same way for `[text_similarity_reranker]`. See [Elastic subscriptions](https://www.elastic.co/subscriptions) for which tiers include it. To evaluate on a test cluster, start a trial with `POST _license/start_trial?acknowledge=true`.
- **A running container** that Elasticsearch can reach over the network. [Quick Start](Quick-Start) gets one up. The URL you register is resolved from the Elasticsearch nodes, not from your laptop, so `localhost` only works when both run on the same host outside containers.

## Register an embedding endpoint

The requests below are in Kibana Dev Tools format.

```json
PUT _inference/text_embedding/jina-embed
{
  "service": "openai",
  "service_settings": {
    "url": "http://jina-host:8080/v1/embeddings",
    "model_id": "jina-embeddings-v5-text-nano",
    "api_key": "not-needed"
  }
}
```

`model_id` is the model the container serves; `GET /health` on the container reports it. `api_key` is required by the Elasticsearch schema and ignored by jina-on-prem. Elasticsearch probes the endpoint while registering and fills in `dimensions` and `similarity` itself, so the response shows the model's output width.

> **Pick a URL you can keep.** `_update` cannot change the `url`, and an endpoint that a `semantic_text` field uses can only be deleted with `?force=true`. To move it, force-delete it and register the same ID again with the new URL. The mapping refers to the ID, so existing documents and searches keep working as long as the model is the same, but writes fail until the endpoint is back. Registering a stable name from the start avoids this: a DNS name, or the load balancer from [More than one replica](#more-than-one-replica), even while only one container sits behind it.

Check it:

```json
POST _inference/text_embedding/jina-embed
{
  "input": ["Hello world"]
}
```

## Index and search with `semantic_text`

Map a field to the endpoint. Elasticsearch embeds the field on every write, chunking long text first, and embeds the query text on every search:

```json
PUT docs
{
  "mappings": {
    "properties": {
      "content": {
        "type": "semantic_text",
        "inference_id": "jina-embed"
      }
    }
  }
}

POST docs/_doc?refresh=wait_for
{
  "content": "Paris is the capital of France."
}

GET docs/_search
{
  "query": {
    "match": {
      "content": "Which city is the French capital?"
    }
  }
}
```

A plain `match` query on a `semantic_text` field runs as a semantic query.

## Register a reranker and rerank results

```json
PUT _inference/rerank/jina-rerank
{
  "service": "cohere",
  "service_settings": {
    "url": "http://jina-rerank-host:8080/v1/rerank",
    "model_id": "jina-reranker-v3.5",
    "api_key": "not-needed"
  }
}
```

Always set `url`: without it, the `cohere` service sends documents to Cohere's hosted API. Newer Elasticsearch versions take only the scheme, host and port from it and always call `/v2/rerank` (what 9.5.3 does), while 9.0.8 calls the path you register. jina-on-prem serves both, so either works; a proxy in front has to pass `/v2/rerank` through as well.

Wrap any retriever in a `text_similarity_reranker`. Elasticsearch takes the top `rank_window_size` hits from the inner retriever, sends them to the reranker, and returns them in the reranker's order:

```json
GET docs/_search
{
  "retriever": {
    "text_similarity_reranker": {
      "retriever": {
        "standard": {
          "query": {
            "match": { "content": "Which city is the French capital?" }
          }
        }
      },
      "field": "content",
      "inference_id": "jina-rerank",
      "inference_text": "Which city is the French capital?",
      "rank_window_size": 50
    }
  }
}
```

The reranker scores every document in the window on every search, so `rank_window_size` is the main cost knob. Reranker throughput also drops sharply as documents get longer; see [Sizing & Hardware](Sizing-And-Hardware).

## Rate limit on the Elasticsearch side

Elasticsearch throttles each inference endpoint with its own `rate_limit`, and applies a default when you set none: 3000 requests per minute for `openai` endpoints and 10000 for `cohere` on both 9.0.8 and 9.5.3. The value in force is in the response to the registration `PUT`, and in `GET _inference/text_embedding/jina-embed`. If you run replicas to raise throughput, raise the limit too, or Elasticsearch caps the traffic before it reaches them. Set it when you register:

```json
PUT _inference/text_embedding/jina-embed
{
  "service": "openai",
  "service_settings": {
    "url": "http://jina-host:8080/v1/embeddings",
    "model_id": "jina-embeddings-v5-text-nano",
    "api_key": "not-needed",
    "rate_limit": { "requests_per_minute": 60000 }
  }
}
```

On 9.5.3 you can also change it later with `PUT _inference/text_embedding/jina-embed/_update` and `{"service_settings": {"rate_limit": {"requests_per_minute": 60000}}}`. On 9.0.8 that call returned success without applying the new limit, and reset the endpoint's chunking settings, so set the limit at registration there.

## More than one replica

An inference endpoint stores exactly one URL, and Elasticsearch does no client-side load balancing: retries go back to the same URL. To spread load over several containers, put a load balancer in front and register its address.

Use a proxy that picks a backend for each request, such as nginx, Envoy or HAProxy in HTTP mode. Elasticsearch keeps HTTP connections open and reuses them, so a balancer that picks a backend once per TCP connection may keep sending most of the traffic to the same replica.

### Docker Compose

[`docker-compose.replicas.yml`](https://github.com/jina-ai/jina-on-prem/blob/main/docker-compose.replicas.yml) runs two replicas of one model behind nginx, configured by [`nginx-replicas.conf`](https://github.com/jina-ai/jina-on-prem/blob/main/nginx-replicas.conf), with a single published port:

```bash
MODEL=jina-embeddings-v5-text-nano RUNTIME=cpu \
  docker compose -f docker-compose.replicas.yml up -d
curl http://localhost:8080/health
```

Then register `http://<compose-host>:8080/v1/embeddings` as above. nginx starts only once both replicas pass the health check built into the image. A request that fails to connect, times out, or gets a `503` from a replica is retried on the other one, and after three such failures the replica is left out of rotation for 30 seconds.

nginx uses `least_conn`, which sends each request to the replica with the fewest requests in flight. Request time varies a lot with input length, most of all for rerankers, and round-robin would keep handing new work to a replica still busy with a slow batch.

On this setup, requests sent from Elasticsearch split evenly across both replicas. So did 20 requests sent over a single reused keep-alive connection, because nginx chooses a replica per request.

To add a replica, copy the `jina-2` service and add its `server` line to `nginx-replicas.conf`. On GPU, give each replica its own card with a different `device_ids` entry; two model servers on one GPU compete for the same memory and compute.

### Kubernetes

[`k8s/jina-on-prem.yaml`](https://github.com/jina-ai/jina-on-prem/blob/main/k8s/jina-on-prem.yaml) creates a Deployment with two replicas, a `ClusterIP` Service, an autoscaler and an optional nginx Ingress. What to register depends on where Elasticsearch runs:

| Elasticsearch runs | Register | Balanced |
|---|---|---|
| Outside the cluster | the Ingress host, `http://jina-embed.internal/v1/embeddings` in the example manifest | per request (nginx Ingress) |
| In the same cluster | the Service, `http://jina-embed.jina-on-prem.svc.cluster.local:8080/v1/embeddings` | per connection (kube-proxy) |

A plain Service is balanced by kube-proxy, which picks a pod when a connection opens and keeps it for the life of that connection. Because Elasticsearch reuses connections, traffic may concentrate on a few pods. We have not measured this. If you need an even spread with Elasticsearch in the same cluster, route it through the Ingress or a service mesh that balances at L7.

`kubectl port-forward` to the Service always lands on one pod. Use it to check that a pod answers, not to check load balancing.

## Troubleshooting

- **`current license is non-compliant for [inference]`**: the cluster's license does not include the inference API. See [Before you start](#before-you-start).
- **`Inference endpoint not found`** when indexing: the `PUT _inference/...` failed or was never run. Registration errors are returned by the `PUT` itself, and nothing retries it.
- **`Inference endpoint [...] already exists`**: registration only creates. To change the URL, see [Pick a URL you can keep](#register-an-embedding-endpoint).
- **Registration fails with `... status code for request from inference entity id [...] status [400]`**: `model_id` does not match the model the container serves. Each image serves one model; copy the name from `GET /health` on the container.
- **Registration fails with a connection error**: the URL is resolved from the Elasticsearch nodes. Test it from there, for example `curl http://jina-host:8080/health` on an Elasticsearch node.
- **Throughput stops rising when you add replicas**: check the endpoint's `rate_limit` (see [above](#rate-limit-on-the-elasticsearch-side)), then check that requests reach every replica (`docker logs` on each).
- **`413` through nginx**: the request body is over the proxy's limit. `nginx-replicas.conf` sets `client_max_body_size 32m`; raise it if you send larger multimodal batches.

More in [Troubleshooting](Troubleshooting).
