import { useCallback, useState } from 'react';

type Props = {
  disabled?: boolean;
  accept?: string;
  hint?: string;
  onFile: (file: File) => void;
};

export default function FileDropZone({ disabled, accept, hint, onFile }: Props) {
  const [drag, setDrag] = useState(false);

  const handleFiles = useCallback(
    (files: FileList | null) => {
      if (!files?.length || disabled) return;
      onFile(files[0]);
    },
    [disabled, onFile]
  );

  return (
    <div
      role="button"
      tabIndex={0}
      onDragEnter={(e) => {
        e.preventDefault();
        e.stopPropagation();
        if (!disabled) setDrag(true);
      }}
      onDragOver={(e) => {
        e.preventDefault();
        e.stopPropagation();
        if (!disabled) setDrag(true);
      }}
      onDragLeave={(e) => {
        e.preventDefault();
        e.stopPropagation();
        setDrag(false);
      }}
      onDrop={(e) => {
        e.preventDefault();
        e.stopPropagation();
        setDrag(false);
        if (!disabled) {
          console.log('File dropped:', e.dataTransfer.files);
          handleFiles(e.dataTransfer.files);
        }
      }}
      className={`rounded-xl border-2 border-dashed px-6 py-10 text-center transition-colors ${
        drag ? 'border-primary-500 bg-primary-950/30' : 'border-surface-600 bg-surface-900/50'
      } ${disabled ? 'opacity-50 pointer-events-none' : 'cursor-pointer hover:border-surface-500'}`}
      onClick={() => {
        if (disabled) return;
        const input = document.createElement('input');
        input.type = 'file';
        if (accept) input.accept = accept;
        input.onchange = () => {
          console.log('File selected via picker:', input.files);
          handleFiles(input.files);
        };
        input.click();
      }}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          (e.target as HTMLElement).click();
        }
      }}
    >
      <div className="text-3xl mb-2">📤</div>
      <p className="text-white text-sm font-medium">Drop a file here or click to browse</p>
      {hint ? <p className="text-surface-500 text-xs mt-2">{hint}</p> : null}
    </div>
  );
}
