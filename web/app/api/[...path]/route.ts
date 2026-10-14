import { NextRequest, NextResponse } from "next/server";

const base = process.env.API_BASE_URL || "http://localhost:8000";
async function proxy(request: NextRequest, context: { params: Promise<{ path: string[] }> }) {
  const { path } = await context.params;
  const target = `${base}/api/${path.map(encodeURIComponent).join("/")}${request.nextUrl.search}`;
  const headers = new Headers();
  const contentType = request.headers.get("content-type");
  if (contentType) headers.set("content-type", contentType);
  const response = await fetch(target, { method: request.method, headers, body: request.method === "GET" || request.method === "HEAD" ? undefined : await request.arrayBuffer(), cache: "no-store" });
  return new NextResponse(response.body, { status: response.status, headers: { "content-type": response.headers.get("content-type") || "application/json", "content-disposition": response.headers.get("content-disposition") || "" } });
}
export const GET = proxy;
export const POST = proxy;
