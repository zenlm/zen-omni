export function Logo({ size = 24, className = "", variant = "default" }: { size?: number; className?: string; variant?: "default" | "white" }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 100 100"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
    >
      {/* Zen enso circle */}
      <circle cx="50" cy="50" r="45" fill="none" stroke="currentColor" strokeWidth="3" opacity="0.8" />
      <path
        d="M50 10 Q70 30 50 50 Q30 70 50 90"
        fill="none"
        stroke="currentColor"
        strokeWidth="4"
        strokeLinecap="round"
      />
    </svg>
  );
}
