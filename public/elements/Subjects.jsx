import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ChevronRight, ChevronDown, FolderPlus, Upload, Trash2, FileText, Plus } from "lucide-react";

const TYPES = ["notes", "exams", "code", "admin"];
const LABEL = { notes: "Notes", exams: "Exams", code: "Code", admin: "Admin" };

function Counts({ counts }) {
  const parts = TYPES.filter((t) => counts[t]).map((t) => `${LABEL[t]} ${counts[t]}`);
  return <span className="text-xs text-muted-foreground">{parts.length ? parts.join(" · ") : "Empty"}</span>;
}

// two-step delete: first click arms, second click confirms (no browser dialogs)
function DeleteButton({ onConfirm, label }) {
  const [armed, setArmed] = useState(false);
  return armed ? (
    <span className="flex items-center gap-1">
      <Button size="sm" variant="destructive" className="h-7 px-2 text-xs" onClick={onConfirm}>Delete {label}</Button>
      <Button size="sm" variant="ghost" className="h-7 px-2 text-xs" onClick={() => setArmed(false)}>Cancel</Button>
    </span>
  ) : (
    <Button size="icon" variant="ghost" className="h-7 w-7 text-muted-foreground hover:text-destructive"
            title={`Delete ${label}`} onClick={() => setArmed(true)}>
      <Trash2 className="h-3.5 w-3.5" />
    </Button>
  );
}

function Subject({ uni, subject }) {
  const [open, setOpen] = useState(false);
  const [picking, setPicking] = useState(false);
  return (
    <div className="border-t border-border first:border-t-0">
      <div className="flex items-center gap-2 py-2.5">
        <button className="flex flex-1 items-center gap-2 text-left" onClick={() => setOpen(!open)}>
          {open ? <ChevronDown className="h-4 w-4 text-muted-foreground" /> : <ChevronRight className="h-4 w-4 text-muted-foreground" />}
          <span className="font-medium">{subject.name}</span>
          <Counts counts={subject.counts} />
        </button>
        {picking ? (
          <span className="flex items-center gap-1">
            {TYPES.map((t) => (
              <Button key={t} size="sm" variant="secondary" className="h-7 px-2.5 text-xs"
                      onClick={() => { setPicking(false); callAction({ name: "subjects_upload", payload: { uni, course: subject.name, doc_type: t } }); }}>
                {LABEL[t]}
              </Button>
            ))}
            <Button size="sm" variant="ghost" className="h-7 px-2 text-xs" onClick={() => setPicking(false)}>Cancel</Button>
          </span>
        ) : (
          <Button size="sm" variant="outline" className="h-7 gap-1.5 px-2.5 text-xs" onClick={() => setPicking(true)}>
            <Upload className="h-3.5 w-3.5" /> Upload
          </Button>
        )}
        <DeleteButton label={subject.name}
                      onConfirm={() => callAction({ name: "subjects_delete_subject", payload: { uni, course: subject.name } })} />
      </div>
      {open && (
        <ul className="mb-2 ml-6 space-y-0.5">
          {subject.files.length === 0 && <li className="text-xs text-muted-foreground">No files yet — use Upload.</li>}
          {subject.files.map((f) => (
            <li key={f.path} className="flex items-center gap-2 text-sm">
              <FileText className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
              <span className="flex-1 truncate">{f.name}</span>
              <span className="text-xs text-muted-foreground">{LABEL[f.type] || f.type}</span>
              <DeleteButton label="file" onConfirm={() => callAction({ name: "subjects_delete_file", payload: { path: f.path } })} />
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function AddSubject({ defaultUni }) {
  const [open, setOpen] = useState(false);
  const [uni, setUni] = useState(defaultUni || "");
  const [course, setCourse] = useState("");
  if (!open)
    return (
      <Button size="sm" className="h-8 gap-1.5" onClick={() => setOpen(true)}>
        <FolderPlus className="h-4 w-4" /> Add subject
      </Button>
    );
  const submit = () => {
    if (!uni.trim() || !course.trim()) return;
    callAction({ name: "subjects_add", payload: { uni: uni.trim(), course: course.trim() } });
    setCourse(""); setOpen(false);
  };
  return (
    <div className="flex w-full items-center gap-2 pt-1">
      <Input className="h-8 w-32" placeholder="University" value={uni} onChange={(e) => setUni(e.target.value)} />
      <Input className="h-8 flex-1" placeholder="Subject, e.g. INF1010" value={course}
             onChange={(e) => setCourse(e.target.value)} onKeyDown={(e) => e.key === "Enter" && submit()} />
      <Button size="sm" className="h-8 gap-1" onClick={submit}><Plus className="h-4 w-4" /> Create</Button>
      <Button size="sm" variant="ghost" className="h-8" onClick={() => setOpen(false)}>Cancel</Button>
    </div>
  );
}

export default function Subjects() {
  const tree = props.tree || [];
  return (
    <div className="w-full max-w-2xl rounded-xl border border-border bg-card p-5">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <div className="min-w-[12rem]">
          <div className="text-base font-medium">Subjects</div>
          <div className="text-xs text-muted-foreground">Your notes, by university and subject</div>
        </div>
        <AddSubject defaultUni={tree[0] && tree[0].uni} />
      </div>
      {props.status && <div className="mb-3 rounded-md bg-accent px-3 py-2 text-xs text-accent-foreground">{props.status}</div>}
      {tree.length === 0 && <div className="py-6 text-center text-sm text-muted-foreground">No subjects yet. Add one to get started.</div>}
      {tree.map((u) => (
        <div key={u.uni} className="mb-4 last:mb-0">
          <div className="mb-1 text-xs font-medium uppercase tracking-wider text-primary">{u.uni}</div>
          <div className="rounded-lg border border-border px-3">
            {u.subjects.length === 0 && <div className="py-2.5 text-xs text-muted-foreground">No subjects</div>}
            {u.subjects.map((s) => <Subject key={s.name} uni={u.uni} subject={s} />)}
          </div>
          {u.files.length > 0 && (
            <div className="mt-1 text-xs text-muted-foreground">+ {u.files.length} university-wide file(s)</div>
          )}
        </div>
      ))}
    </div>
  );
}
