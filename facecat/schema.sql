create extension if not exists vector;

create table if not exists roots (
    id bigserial primary key,
    path text not null unique,
    enabled boolean not null default true,
    created_at timestamptz not null default now(),
    last_indexed_at timestamptz
);

create table if not exists files (
    id bigserial primary key,
    root_id bigint not null references roots(id) on delete cascade,
    rel_path text not null,
    abs_path text not null unique,
    file_size bigint not null,
    mtime timestamptz not null,
    width int,
    height int,
    exif jsonb not null default '{}'::jsonb,
    file_thumb_rel text,
    indexed_at timestamptz,
    last_seen_at timestamptz not null default now(),
    is_deleted boolean not null default false,
    unique (root_id, rel_path)
);

create index if not exists idx_files_root_id on files(root_id);
create index if not exists idx_files_is_deleted on files(is_deleted);

create table if not exists faces (
    id bigserial primary key,
    file_id bigint not null references files(id) on delete cascade,
    face_index int not null,
    bbox_x1 int not null,
    bbox_y1 int not null,
    bbox_x2 int not null,
    bbox_y2 int not null,
    det_score real,
    quality_score real,
    embedding vector(512) not null,
    face_thumb_rel text,
    invalidated boolean not null default false,
    invalidated_reason text,
    created_at timestamptz not null default now(),
    unique (file_id, face_index)
);

create index if not exists idx_faces_file_id on faces(file_id);
create index if not exists idx_faces_invalidated on faces(invalidated);
create index if not exists idx_faces_embedding_hnsw
on faces using hnsw (embedding vector_cosine_ops);

create table if not exists face_groups (
    id bigserial primary key,
    label text,
    rep_face_id bigint references faces(id) on delete set null,
    member_count int not null default 0,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table if not exists face_group_members (
    group_id bigint not null references face_groups(id) on delete cascade,
    face_id bigint not null unique references faces(id) on delete cascade,
    score_to_rep real,
    created_at timestamptz not null default now(),
    primary key (group_id, face_id)
);

create index if not exists idx_face_group_members_group_id on face_group_members(group_id);

create table if not exists jobs (
    id bigserial primary key,
    job_type text not null,
    status text not null,
    started_at timestamptz not null default now(),
    finished_at timestamptz,
    stats jsonb not null default '{}'::jsonb,
    error text
);

-- kNN edges computed during group rebuild (persisted per block so an
-- interrupted rebuild can resume). Cleared when a new job starts.
create table if not exists face_group_edges (
    job_id bigint references jobs(id) on delete cascade,
    face_id bigint not null references faces(id) on delete cascade,
    nbr_face_id bigint not null references faces(id) on delete cascade,
    similarity float8 not null,
    primary key (face_id, nbr_face_id)
);

create index if not exists idx_group_edges_nbr on face_group_edges (nbr_face_id);
