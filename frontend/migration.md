# Moving the AEPO frontend to another device

**Short answer:** Unzip alone is **not** enough. You need **Node.js**, then **`npm install`**, then **`npm run dev`** (or build). The dashboard talks to a backend over HTTP; without it, API calls fail.

---

## What the portable zip contains

If you use the recommended archive (source only, no `node_modules`, no `.next`):

- Application source, config, and lockfile (if present)
- You **must** run `npm install` on the new machine to recreate dependencies

If someone zips **with** `node_modules` included: it can still break on another OS or CPU (native addons), and it is huge—prefer a source-only zip.

---

## Prerequisites on the new device

| Requirement | Notes |
|-------------|--------|
| **Node.js** | Use an LTS version compatible with Next.js 14 (e.g. **Node 18.x or 20.x**). Check with `node -v`. |
| **npm** | Ships with Node (`npm -v`). |
| **Backend (optional but needed for live data)** | `next.config.mjs` rewrites `/api/*` to `http://localhost:7860/*`. Run your FastAPI / AEPO server on **port 7860**, or change the rewrite target to match where the API actually runs. |

No `.env` file is required for the default setup; the API base is `/api` and the rewrite points at localhost.

---

## Steps after copying the zip

1. **Unzip** into a folder of your choice (e.g. `~/projects/aepo-dashboard`).

2. **Open a terminal** in the unzipped `frontend` directory (the folder that contains `package.json`).

3. **Install dependencies:**

   ```bash
   npm install
   ```

4. **Development server:**

   ```bash
   npm run dev
   ```

   Open [http://localhost:3000](http://localhost:3000).

5. **Production-style run** (after install):

   ```bash
   npm run build
   npm run start
   ```

6. **If the UI loads but API errors appear:** start the backend on the host/port expected by `next.config.mjs` (default **7860**), or edit `destination` in the `rewrites` section to your API URL and restart Next.

---

## Optional checks

```bash
npm run lint
```

---

## Regenerating the portable zip (on the machine that has the repo)

From the **repository root** (parent of `frontend`), excluding heavy or machine-local folders:

```bash
zip -r frontend-portable.zip frontend \
  -x "frontend/node_modules/*" \
  -x "frontend/.next/*"
```

If your `zip` does not exclude nested paths as expected, use a clean copy:

```bash
rsync -a --exclude node_modules --exclude .next frontend/ /tmp/frontend-export/
cd /tmp && zip -r /path/to/frontend-portable.zip frontend
```

---

## Summary

| Step | Required? |
|------|-----------|
| Unzip | Yes |
| Install Node.js | Yes |
| `npm install` | Yes (if zip excluded `node_modules`) |
| `npm run dev` or build/start | Yes, to run the app |
| Backend on :7860 (or rewrites updated) | Yes, for a working dashboard against the real API |
