# Docker + Neo4j Setup

Steps to get a local Neo4j instance running via Docker, for testing
`graph/neo4j_client.py` and (later) `graph/loader.py` against a real
database instead of guessing at Cypher blind.

---

## 1. Start Docker Desktop

```bash
open -a "Docker Desktop"
```

Watch the menu bar whale icon — animates while booting, settles solid when
ready. First launch after a while can take 30-60s.

Confirm the daemon is actually up (not just the UI):

```bash
docker info
```

Look for a `Server:` block in the output. If you only see a `Client:` block
and a `Cannot connect to the Docker daemon` error, the daemon isn't up yet —
wait longer, or see Troubleshooting below.

## 2. Run the Neo4j container

```bash
docker run -d \
  --name graphrag-neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/<pick-a-password> \
  neo4j:latest
```

What each flag does:
- `-d` — run in the background (detached), don't block the terminal
- `--name graphrag-neo4j` — a name to refer to this container by, instead of
  its generated ID
- `-p 7474:7474` — expose the Neo4j Browser (web UI) on `localhost:7474`
- `-p 7687:7687` — expose the Bolt port on `localhost:7687`, the protocol
  `neo4j_client.py`'s driver actually connects over
- `-e NEO4J_AUTH=neo4j/<password>` — sets the initial username/password at
  first boot. Username `neo4j` is fixed on the free/community edition; pick
  a real password, not a placeholder.
- `neo4j:latest` — the image to run. First run downloads it (a few hundred
  MB); later runs reuse the cached image.

## 3. Wait for it to actually finish booting

```bash
docker logs -f graphrag-neo4j
```

Tail the logs until you see a line like `Started.`, then `Ctrl+C` to stop
tailing (the container keeps running in the background — `Ctrl+C` only
exits the log view).

## 4. Point the project's env vars at it

Matches what `src/graph/neo4j_client.py` reads by default:

```bash
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=<the password from step 2>
```

## 5. Sanity-check in the browser (optional but easy)

Open `http://localhost:7474`, log in with the same username/password. If
that works, the instance itself is healthy before anything in Python
touches it.

## 6. Useful commands going forward

```bash
docker ps --filter name=graphrag-neo4j     # is it running?
docker stop graphrag-neo4j                 # stop (keeps data)
docker start graphrag-neo4j                # resume a stopped container
docker rm -f graphrag-neo4j                # remove entirely (data lost
                                            # unless a volume was mounted)
```

Data isn't persisted across `docker rm` in this setup (no volume mounted) —
fine for dev/testing, not for anything you want to keep. Add
`-v graphrag-neo4j-data:/data` to the `docker run` command if persistence
across container recreation matters later.

---

## Troubleshooting

**`Cannot connect to the Docker daemon` even after `open -a Docker Desktop`:**
Check for a stuck/dormant process:
```bash
ps aux | grep -i "Docker Desktop" | grep -v grep
```
If it's been running for a long time with no actual daemon responding, it's
likely stuck. Force-quit and relaunch fresh:
```bash
osascript -e 'quit app "Docker Desktop"'
sleep 3
pkill -9 -f "Docker Desktop"
open -a "Docker Desktop"
```

**An installer/updater process appears instead of the normal app**
(`.../install --launch-args=...`): Docker's auto-update triggered. Check
your screen for an installer window or an admin-password prompt — it needs
GUI interaction to complete. Once it finishes, Docker Desktop should launch
normally.

**Container starts but `docker logs` never shows `Started.`:** give it more
time (30s+) — Neo4j does real startup work (index rebuilding, etc.) on
first boot, not just a process launch.
