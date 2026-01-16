export function Button({ children, onClick }) {
  return (
    <button
      onClick={onClick}
      style={{
        padding: "8px 16px",
        borderRadius: 6,
        background: "#007bff",
        color: "#fff",
        border: "none",
        cursor: "pointer",
      }}
    >
      {children}
    </button>
  );
}
