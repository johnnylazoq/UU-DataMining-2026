import Dashboard from "../../dashboard";
export function generateStaticParams() { return ["clustering","bluetooth","gatherings","episodes","cohorts","associations","anomalies"].map((slug)=>({slug})); }
export default async function Page({ params }: { params: Promise<{slug:string}> }) { const {slug}=await params; return <Dashboard slug={slug}/>; }
