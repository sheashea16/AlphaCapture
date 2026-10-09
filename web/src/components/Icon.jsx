const paths = {
  arrow: "M5 12h14m-6-6 6 6-6 6",
  external: "M14 4h6v6M20 4 10 14M10 4H4v16h16v-6",
  reset: "M4 10a8 8 0 1 1 1 8M4 4v6h6",
  play: "m8 5 11 7-11 7Z",
  pause: "M8 5v14M16 5v14",
  close: "m6 6 12 12M6 18 18 6",
  undo: "m9 5-5 5 5 5M4 10h9a6 6 0 0 1 0 12",
  book: "M12 5C9 3 5 3 3 4v15c3-1 6-1 9 1 3-2 6-2 9-1V4c-2-1-6-1-9 1Zm0 0v15",
  spark: "m12 3 2.5 6.5L21 12l-6.5 2.5L12 21l-2.5-6.5L3 12l6.5-2.5Z",
};
export default function Icon({ name, size = 18 }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d={paths[name] || paths.arrow} />
    </svg>
  );
}
export function Mark() {
  return (
    <svg
      className="brand-mark"
      viewBox="0 0 32 32"
      fill="none"
      aria-hidden="true"
    >
      <path d="m4 26 11-21h4l10 21h-7l-5-11-6 11Z" fill="currentColor" />
      <circle cx="17" cy="25" r="3" fill="currentColor" />
    </svg>
  );
}
