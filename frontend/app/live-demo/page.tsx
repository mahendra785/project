import { LiveDemo } from '@/components/live-demo/live-demo';

export const metadata = {
  title: 'Live demo — PUMS',
  description: 'Live lesion-level joint conformal risk control on real Kvasir-SEG posteriors: pixel CRC vs the joint rule, image by image.',
};

export default function LiveDemoPage() {
  return <LiveDemo />;
}
