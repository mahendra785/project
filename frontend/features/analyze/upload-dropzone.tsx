'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { FileImage, Upload, X } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { bytes, cn } from '@/lib/utils';

const ACCEPTED = ['image/png', 'image/jpeg', 'image/bmp', 'image/tiff', 'application/dicom'];
const ACCEPT_ATTR = '.png,.jpg,.jpeg,.bmp,.tif,.tiff,.dcm,application/dicom';
const MAX_BYTES = 25 * 1024 * 1024;

export function UploadDropzone({
  file,
  onFileChange,
  disabled,
}: {
  file: File | null;
  onFileChange: (file: File | null) => void;
  disabled?: boolean;
}) {
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const isDicom = Boolean(file && /\.dcm$/i.test(file.name));

  // Derived during render rather than pushed into state by an effect: the URL
  // is a pure function of the file, and setState-in-effect would cause an extra
  // render pass on every selection.
  const previewUrl = useMemo(
    () => (file && !isDicom ? URL.createObjectURL(file) : null),
    [file, isDicom],
  );

  // Object URLs must still be revoked, or the browser leaks the decoded image.
  useEffect(() => {
    if (!previewUrl) return;
    return () => URL.revokeObjectURL(previewUrl);
  }, [previewUrl]);

  const accept = useCallback(
    (candidate: File | undefined) => {
      if (!candidate) return;
      setError(null);

      if (candidate.size > MAX_BYTES) {
        setError(`File is ${bytes(candidate.size)}; the limit is 25 MB.`);
        return;
      }
      // DICOM often arrives with an empty or generic MIME type, so extension
      // is a legitimate fallback — the backend sniffs the magic bytes anyway.
      const isDicomName = /\.dcm$/i.test(candidate.name);
      if (candidate.type && !ACCEPTED.includes(candidate.type) && !isDicomName) {
        setError(`Unsupported type "${candidate.type}". Use PNG, JPEG, TIFF, BMP or DICOM.`);
        return;
      }
      onFileChange(candidate);
    },
    [onFileChange],
  );

  if (file) {
    return (
      <div className="space-y-2">
        <div className="relative overflow-hidden rounded-md border border-border">
          {previewUrl ? (
            // eslint-disable-next-line @next/next/no-img-element -- local object URL
            <img
              src={previewUrl}
              alt="Selected image preview"
              className="checkerboard aspect-square w-full object-contain"
            />
          ) : (
            <div className="flex aspect-square w-full flex-col items-center justify-center gap-2 bg-muted/40 text-muted-foreground">
              <FileImage className="size-8" />
              <p className="text-xs">DICOM preview is rendered after analysis</p>
            </div>
          )}
          <Button
            variant="secondary"
            size="icon-sm"
            className="absolute right-2 top-2 shadow"
            aria-label="Remove image"
            disabled={disabled}
            onClick={() => {
              onFileChange(null);
              if (inputRef.current) inputRef.current.value = '';
            }}
          >
            <X className="size-3.5" />
          </Button>
        </div>
        <div className="flex items-center justify-between gap-2 text-[11px] text-muted-foreground">
          <span className="truncate font-medium text-foreground">{file.name}</span>
          <span className="tabular shrink-0">{bytes(file.size)}</span>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <div
        role="button"
        tabIndex={0}
        aria-label="Upload an image"
        onClick={() => inputRef.current?.click()}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') inputRef.current?.click();
        }}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          accept(e.dataTransfer.files?.[0]);
        }}
        className={cn(
          'flex aspect-square w-full cursor-pointer flex-col items-center justify-center gap-3 rounded-md border-2 border-dashed p-6 text-center transition-colors',
          dragging
            ? 'border-primary bg-primary/5'
            : 'border-border hover:border-primary/50 hover:bg-accent/40',
          disabled && 'pointer-events-none opacity-60',
        )}
      >
        <div className="rounded-full bg-muted p-3 text-muted-foreground">
          <Upload className="size-5" />
        </div>
        <div className="space-y-1">
          <p className="text-sm font-medium">Drop an image, or click to browse</p>
          <p className="text-[11px] text-muted-foreground">
            PNG, JPEG, TIFF, BMP or DICOM · up to 25 MB
          </p>
        </div>
        <p className="max-w-xs text-[10px] leading-relaxed text-muted-foreground">
          Sample phantoms are in <code>data/sample/</code>. They are synthetic —
          not medical images.
        </p>
      </div>

      <input
        ref={inputRef}
        type="file"
        accept={ACCEPT_ATTR}
        className="hidden"
        onChange={(e) => accept(e.target.files?.[0])}
      />

      {error && <p className="text-[11px] text-status-danger">{error}</p>}
    </div>
  );
}
