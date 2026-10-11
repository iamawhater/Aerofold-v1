export function token() {
  return sessionStorage.getItem("aerofold-token") || "";
}
export async function api<T>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  const headers = new Headers(options.headers);
  if (token()) headers.set("Authorization", `Bearer ${token()}`);
  if (options.body && !(options.body instanceof FormData))
    headers.set("Content-Type", "application/json");
  const response = await fetch(`/api${path}`, { ...options, headers });
  if (!response.ok) {
    const body = await response
      .json()
      .catch(() => ({ detail: response.statusText }));
    throw new Error(
      typeof body.detail === "string"
        ? body.detail
        : JSON.stringify(body.detail),
    );
  }
  return response.json();
}
export async function download(runId: string, format: string) {
  const response = await fetch(`/api/runs/${runId}/export?format=${format}`, {
    headers: token() ? { Authorization: `Bearer ${token()}` } : {},
  });
  if (!response.ok) {
    const body = await response.json();
    throw new Error(body.detail);
  }
  const url = URL.createObjectURL(await response.blob());
  const link = document.createElement("a");
  link.href = url;
  link.download = `aerofold-${runId.slice(0, 8)}-${format}.${format === "audit" ? "json" : "csv"}`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
