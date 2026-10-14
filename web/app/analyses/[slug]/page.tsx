import Dashboard from "../../dashboard";
import { notFound } from "next/navigation";
export const dynamicParams = false;
export function generateStaticParams() { return ["clustering","bluetooth","gatherings","episodes","cohorts","associations"].map((slug)=>({slug})); }
export default async function Page({ params }: { params: Promise<{slug:string}> }) { const {slug}=await params; if(!["clustering","bluetooth","gatherings","episodes","cohorts","associations"].includes(slug))notFound(); return <Dashboard slug={slug}/>; }
