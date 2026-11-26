-- Run this in Supabase Dashboard > SQL Editor.
-- One row per annotator per sample. Re-saving the same sample updates that row.

create table if not exists public.annotations (
    annotator_id text not null,
    sample_id integer not null,
    relevance text not null check (relevance in ('True', 'False')),
    answerability text not null check (answerability in ('True', 'False')),
    fluency text not null check (fluency in ('True', 'False')),
    factual_consistency text not null check (factual_consistency in ('True', 'False')),
    updated_at timestamptz not null default now(),
    constraint annotations_annotator_sample_pk primary key (annotator_id, sample_id)
);

-- The Streamlit server connects with the service-role key, which bypasses RLS.
-- Do not put the service-role key in app.py, GitHub, or a client-side app.
alter table public.annotations enable row level security;
