export function Card({ children }) {
  return <div style={{ border: "1px solid #ccc", borderRadius: 8, padding: 16, boxShadow: "0 2px 5px rgba(0,0,0,0.1)", background: "#fff" }}>{children}</div>;
}

export function CardContent({ children }) {
  return <div>{children}</div>;
}
