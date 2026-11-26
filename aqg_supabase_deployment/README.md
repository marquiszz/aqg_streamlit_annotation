# AQG Human Annotation — Streamlit + Supabase

This version keeps the existing 250-sample dataset and answer-matching logic, but moves annotation persistence from local CSV files to Supabase.

## What it does

- All annotators see the same 250 samples.
- Each annotator is selected from a dropdown.
- Each record is keyed by `(annotator_id, sample_id)`, so one annotator cannot overwrite another annotator's row.
- Saving a sample upserts the current annotator's decisions immediately to Supabase.
- Returning to a previously saved sample pre-fills that annotator's decisions.
- Progress and resume position are calculated separately for each selected annotator.
- Each annotator can download their own CSV.
- If `admin_password` is configured, an administrator can unlock an all-annotators CSV export.

## 1. Create the Supabase project

1. Go to https://supabase.com/ and create a project.
2. Open **SQL Editor** in the Supabase dashboard.
3. Open `schema.sql` from this package, copy its contents, and run the SQL.
4. In **Project Settings → API**, copy the Project URL and the `service_role` key.

The app uses the service-role key only on the Streamlit server. Never commit it to GitHub or expose it in browser-side code.

## 2. Configure Streamlit secrets

Create `.streamlit/secrets.toml` (the folder must be named `.streamlit`) using `secrets.toml.example` as a template.

Replace:
- `url` with your Supabase Project URL.
- `service_role_key` with your Supabase service-role key.
- `annotators` with the actual display names or stable IDs, e.g. `["A01", "A02", "A03"]`.
- `admin_password.value` with a strong password. This is optional; omit the entire `[admin_password]` section if you do not want the administrator export panel.

Do not upload `secrets.toml` to GitHub. Add it to `.gitignore`.

## 3. Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 4. Deploy to Streamlit Community Cloud

1. Push `app.py`, `requirements.txt`, and `schema.sql` to a GitHub repository.
2. Ensure `.streamlit/secrets.toml` is excluded from Git. A `.gitignore` is included.
3. Go to https://share.streamlit.io/ and create an app from that repository.
4. Set the main file path to `app.py`.
5. Open the deployed app's **Settings → Secrets** and paste the contents of your local `secrets.toml`.
6. Deploy/reboot the app and share its `.streamlit.app` URL with annotators.

## Important limitations of dropdown identification

A dropdown is an identification mechanism, not secure authentication. Anyone who can open the app can select another annotator's label and potentially submit under that label. This meets the requested simple workflow, but it is not suitable if you need verified identity or strict per-annotator access control. For that, use individual login accounts or unique access codes and enforce identity in the backend.

## Backups

Supabase is now the persistent source of truth; local CSV backups are no longer used. Back up the Supabase table regularly using the Supabase dashboard export or download the all-annotators CSV from the administrator export panel.

## Hugging Face access

The sample datasets are downloaded from public Hugging Face repositories as in the original app. If you later make either repository private, add `HF_TOKEN = "your_token"` as a top-level Streamlit secret.
