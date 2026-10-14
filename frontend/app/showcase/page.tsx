import { Showcase } from '@/components/showcase/showcase';
import { loadShowcaseData } from '@/components/showcase/showcase-data';

// Read the committed result files on every request, so regenerated results
// appear without a rebuild.
export const dynamic = 'force-dynamic';

export const metadata = {
  title: 'Project demo — PUMS',
  description: 'End-to-end walkthrough of lesion-level joint conformal risk control: novelty, live workflow and every recorded result.',
};

export default async function ShowcasePage() {
  const data = await loadShowcaseData();
  return <Showcase data={data} />;
}
