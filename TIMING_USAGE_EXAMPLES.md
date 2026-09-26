# Exemples d'utilisation : Mesure de la durée de traitement

## 1️⃣ Requête API simple

### Requête cURL

```bash
curl -X POST http://localhost:8000/process-ticket \
  -H "Content-Type: application/json" \
  -d '{
    "ticket_id": "TKT_2024_001",
    "module": "Facturation",
    "priorite": "Assistance",
    "description": "Comment puis-je exporter une facture en PDF ?"
  }'
```

### Réponse (formatée)

```json
{
  "ticket_id": "TKT_2024_001",
  "status": "reponse_a_valider",
  "reponse_client": "Vous pouvez exporter une facture en PDF en cliquant sur...",
  "meta_support": {
    "confidence": 0.82,
    "sources": [
      {
        "point_id": "abc123",
        "categorie": "Export/Impression",
        "question": "Comment exporter une facture ?",
        "reponse": "Pour exporter..."
      }
    ],
    "escalade_reason": null,
    "model_used": "gemini-2.5-flash-lite",
    "processing_duration_ms": 487.23,
    "step_timings": [
      {
        "step_name": "scope_check",
        "duration_ms": 0.42
      },
      {
        "step_name": "retrieval_hybrid",
        "duration_ms": 156.78
      },
      {
        "step_name": "reranking",
        "duration_ms": 62.15
      },
      {
        "step_name": "retrieval_total",
        "duration_ms": 219.35
      },
      {
        "step_name": "anonymisation",
        "duration_ms": 8.91
      },
      {
        "step_name": "gemini_generation",
        "duration_ms": 258.97
      }
    ]
  }
}
```

### Extraction des timings (Python)

```python
import requests
import json

response = requests.post(
    "http://localhost:8000/process-ticket",
    json={
        "ticket_id": "TKT_2024_001",
        "module": "Facturation",
        "priorite": "Assistance",
        "description": "Comment puis-je exporter une facture en PDF ?",
    }
)

data = response.json()
meta = data["meta_support"]

print(f"Ticket: {data['ticket_id']}")
print(f"Status: {data['status']}")
print(f"Durée totale: {meta['processing_duration_ms']:.1f}ms")
print("\nDétail par étape:")
for step in meta["step_timings"]:
    print(f"  {step['step_name']:25} : {step['duration_ms']:7.2f}ms")
```

## 2️⃣ Cas d'escalade par score insuffisant

### Requête

```bash
curl -X POST http://localhost:8000/process-ticket \
  -H "Content-Type: application/json" \
  -d '{
    "ticket_id": "TKT_2024_002",
    "module": "Facturation",
    "priorite": "Assistance",
    "description": "Pourquoi le ciel est-il bleu ?"
  }'
```

### Réponse

```json
{
  "ticket_id": "TKT_2024_002",
  "status": "escalade",
  "reponse_client": null,
  "meta_support": {
    "confidence": 0.35,
    "sources": [
      {
        "point_id": "xyz789",
        "categorie": "Paramètres généraux",
        "question": "Comment configurer mes paramètres ?",
        "reponse": "..."
      }
    ],
    "escalade_reason": "score_insuffisant",
    "model_used": null,
    "processing_duration_ms": 224.56,
    "step_timings": [
      {
        "step_name": "scope_check",
        "duration_ms": 0.38
      },
      {
        "step_name": "retrieval_hybrid",
        "duration_ms": 142.31
      },
      {
        "step_name": "reranking",
        "duration_ms": 81.87
      }
    ]
  }
}
```

**Note** : Pas de `retrieval_total`, `anonymisation` ou `gemini_generation` → l'IA n'a pas été appelée (score < 0.60)

## 3️⃣ Cas d'escalade par module hors-scope

### Requête

```bash
curl -X POST http://localhost:8000/process-ticket \
  -H "Content-Type: application/json" \
  -d '{
    "ticket_id": "TKT_2024_003",
    "module": "RessourcesHumaines",
    "priorite": "Assistance",
    "description": "Quelle est la politique de congé maternité ?"
  }'
```

### Réponse

```json
{
  "ticket_id": "TKT_2024_003",
  "status": "escalade",
  "reponse_client": null,
  "meta_support": {
    "confidence": 0.0,
    "sources": [],
    "escalade_reason": "module_hors_scope",
    "model_used": null,
    "processing_duration_ms": 0.67,
    "step_timings": [
      {
        "step_name": "scope_check",
        "duration_ms": 0.67
      }
    ]
  }
}
```

**Note** : Temps minimal (~0.67ms) → seule la pré-qualification scope a été exécutée

## 4️⃣ Analyse comparative (3 tickets)

### Python

```python
import requests
from tabulate import tabulate

tickets = [
    {
        "ticket_id": "TKT_001",
        "module": "Facturation",
        "priorite": "Assistance",
        "description": "Comment exporter une facture ?"
    },
    {
        "ticket_id": "TKT_002",
        "module": "Facturation",
        "priorite": "Assistance",
        "description": "Question hors-scope qui ne trouve rien"
    },
    {
        "ticket_id": "TKT_003",
        "module": "Comptabilité",
        "priorite": "Formation",
        "description": "Comment faire une saisie de journal ?"
    },
]

results = []
for ticket in tickets:
    resp = requests.post("http://localhost:8000/process-ticket", json=ticket)
    data = resp.json()
    meta = data["meta_support"]
    
    total_time = meta["processing_duration_ms"]
    retrieval_time = next(
        (s["duration_ms"] for s in meta["step_timings"] if s["step_name"] == "retrieval_total"),
        0
    )
    gemini_time = next(
        (s["duration_ms"] for s in meta["step_timings"] if s["step_name"] == "gemini_generation"),
        0
    )
    
    results.append({
        "Ticket": ticket["ticket_id"],
        "Status": data["status"],
        "Escalade?": meta["escalade_reason"] or "Non",
        "Total (ms)": f"{total_time:.1f}",
        "Retrieval": f"{retrieval_time:.1f}",
        "Gemini": f"{gemini_time:.1f}",
    })

print(tabulate(results, headers="keys", tablefmt="grid"))
```

### Sortie

```
╒═══════════╤═══════════════════╤═══════════════════╤════════════╤══════════════╤══════════╕
│ Ticket    │ Status            │ Escalade?         │ Total (ms) │ Retrieval    │ Gemini   │
╞═══════════╪═══════════════════╪═══════════════════╪════════════╪══════════════╪══════════╡
│ TKT_001   │ reponse_a_valider │ Non               │ 487.2     │ 219.4        │ 259.0    │
│ TKT_002   │ escalade          │ score_insuffisant │ 224.6     │ 224.6        │ 0.0      │
│ TKT_003   │ escalade          │ module_hors_scope │ 0.7       │ 0.0          │ 0.0      │
╘═══════════╧═══════════════════╧═══════════════════╧════════════╧══════════════╧══════════╛
```

## 5️⃣ Dashboard simple (HTML)

```html
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <title>Timing Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .metric { background: #f5f5f5; padding: 15px; border-radius: 8px; }
        .metric h3 { margin: 0; font-size: 14px; color: #666; }
        .metric .value { font-size: 32px; font-weight: bold; color: #333; margin: 10px 0; }
        .metric .unit { font-size: 12px; color: #999; }
        canvas { max-width: 100%; }
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Timing Dashboard - Traitement des tickets</h1>
        
        <div class="metrics">
            <div class="metric">
                <h3>Durée moyenne</h3>
                <div class="value" id="avg-time">-</div>
                <div class="unit">millisecondes</div>
            </div>
            <div class="metric">
                <h3>Tickets traités</h3>
                <div class="value" id="count">0</div>
                <div class="unit">total</div>
            </div>
            <div class="metric">
                <h3>P95 (95e percentile)</h3>
                <div class="value" id="p95-time">-</div>
                <div class="unit">millisecondes</div>
            </div>
        </div>
        
        <div style="background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
            <h2>Distribution des durées</h2>
            <canvas id="timingChart"></canvas>
        </div>
        
        <div style="background: white; padding: 20px; border-radius: 8px;">
            <h2>Breakdown par étape (moyenne)</h2>
            <canvas id="stepsChart"></canvas>
        </div>
    </div>

    <script>
        // Récupérer les données de timings depuis votre backend
        async function loadTimings() {
            const response = await fetch('/api/timings-stats');
            const data = await response.json();
            
            // Mettre à jour les métriques
            document.getElementById('avg-time').textContent = 
                Math.round(data.average_duration_ms);
            document.getElementById('count').textContent = data.total_count;
            document.getElementById('p95-time').textContent = 
                Math.round(data.p95_duration_ms);
            
            // Graphique de distribution
            new Chart(document.getElementById('timingChart'), {
                type: 'bar',
                data: {
                    labels: data.histogram.labels,
                    datasets: [{
                        label: 'Nombre de tickets',
                        data: data.histogram.counts,
                        backgroundColor: '#4CAF50'
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        title: {
                            display: true,
                            text: 'Distribution des durées de traitement'
                        }
                    }
                }
            });
            
            // Graphique des étapes
            new Chart(document.getElementById('stepsChart'), {
                type: 'doughnut',
                data: {
                    labels: data.steps.names,
                    datasets: [{
                        data: data.steps.durations,
                        backgroundColor: [
                            '#FF6384', '#36A2EB', '#FFCE56', 
                            '#4BC0C0', '#9966FF', '#FF9F40'
                        ]
                    }]
                }
            });
        }
        
        // Charger au démarrage et rafraîchir toutes les 10 secondes
        loadTimings();
        setInterval(loadTimings, 10000);
    </script>
</body>
</html>
```

## 6️⃣ Intégration Prometheus (optionnel)

```python
from prometheus_client import Counter, Histogram, generate_latest
from fastapi import Response

# Métriques Prometheus
processing_duration = Histogram(
    'support_ia_processing_duration_ms',
    'Durée de traitement en millisecondes',
    buckets=(10, 50, 100, 200, 500, 1000, 2000)
)

retrieval_duration = Histogram(
    'support_ia_retrieval_duration_ms',
    'Durée du retrieval RAG',
    buckets=(10, 50, 100, 200, 500)
)

gemini_duration = Histogram(
    'support_ia_gemini_duration_ms',
    'Durée de l\'appel Gemini',
    buckets=(100, 200, 500, 1000, 2000)
)

tickets_processed = Counter(
    'support_ia_tickets_processed',
    'Nombre de tickets traités',
    ['status', 'escalade_reason']
)

# Dans le endpoint /process-ticket
@app.post("/process-ticket")
def process_ticket(ticket: TicketInput, request: Request) -> AgentResponse:
    # ... code existant ...
    response = agent4.traiter(ticket)
    
    # Enregistrer les métriques
    processing_duration.observe(
        response.meta_support.processing_duration_ms
    )
    
    for step in response.meta_support.step_timings:
        if step.step_name == "retrieval_total":
            retrieval_duration.observe(step.duration_ms)
        elif step.step_name == "gemini_generation":
            gemini_duration.observe(step.duration_ms)
    
    tickets_processed.labels(
        status=response.status,
        escalade_reason=response.meta_support.escalade_reason or "none"
    ).inc()
    
    return response

# Endpoint Prometheus
@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type="text/plain")
```

## 7️⃣ Logs structurés

Les timings sont aussi disponibles dans les logs INFO :

```
INFO:ai_engine.api.main:Ticket TKT_2024_001 traité en 487.2 ms (status=reponse_a_valider)
INFO:ai_engine.api.main:Ticket TKT_2024_002 traité en 224.5 ms (status=escalade)
INFO:ai_engine.api.main:Ticket TKT_2024_003 traité en 0.8 ms (status=escalade)
ERROR:ai_engine.api.main:Erreur technique inattendue (TimeoutError) lors du traitement du ticket TKT_2024_004 (après 2156.3 ms)
```

## 📈 Bonnes pratiques

✅ **À faire** :
- Monitorer les p95/p99 des durées globales
- Alerter si `retrieval_hybrid > 400ms` (Qdrant lent)
- Alerter si `gemini_generation > 1000ms` (réseau/quota)
- Tracker les timings par module pour comparaison
- Exporter les timings vers Prometheus/Grafana

❌ **À éviter** :
- N'ignorer pas les timings des escalades (diagnostic utile)
- Ne pas cacher les timings au support (transparence)
- Ne pas ajouter de timings en production sauf si justified

## 🔧 Configuration

Tous les timings sont inclus par défaut. Aucune configuration requise.
