import {
  Activity,
  Boxes,
  BrainCircuit,
  FlaskConical,
  GaugeCircle,
  LayoutDashboard,
  Radar,
  ScanSearch,
  Stethoscope,
} from 'lucide-react';

export interface NavItem {
  href: string;
  label: string;
  icon: typeof Activity;
  description: string;
  group: 'Analysis' | 'Research' | 'System';
}

export const NAV_ITEMS: NavItem[] = [
  {
    href: '/dashboard',
    label: 'Dashboard',
    icon: LayoutDashboard,
    description: 'Aggregate statistics over analyses run on this instance',
    group: 'Analysis',
  },
  {
    href: '/analyze',
    label: 'Analyze',
    icon: ScanSearch,
    description: 'Upload an image and run the full PUMS pipeline',
    group: 'Analysis',
  },
  {
    href: '/uncertainty',
    label: 'Uncertainty',
    icon: BrainCircuit,
    description: 'Hierarchical pixel, region and global uncertainty',
    group: 'Analysis',
  },
  {
    href: '/decision-support',
    label: 'Decision support',
    icon: Stethoscope,
    description: 'Risk stratification and referral prioritisation prototype',
    group: 'Analysis',
  },
  {
    href: '/calibration',
    label: 'Calibration',
    icon: GaugeCircle,
    description: 'Conformal risk control, ECE and reliability diagrams',
    group: 'Research',
  },
  {
    href: '/distribution-shift',
    label: 'Distribution shift',
    icon: Radar,
    description: 'OOD detection against the reference domain',
    group: 'Research',
  },
  {
    href: '/evaluation',
    label: 'Evaluation',
    icon: Activity,
    description: 'Metric implementations and session measurements',
    group: 'Research',
  },
  {
    href: '/research',
    label: 'Research status',
    icon: FlaskConical,
    description: 'What is implemented, what is planned, and the literature map',
    group: 'Research',
  },
  {
    href: '/models',
    label: 'Models',
    icon: Boxes,
    description: 'Model adapters, provenance and wiring',
    group: 'System',
  },
];

export const NAV_GROUPS: Array<NavItem['group']> = ['Analysis', 'Research', 'System'];

export function findNavItem(pathname: string): NavItem | undefined {
  return NAV_ITEMS.find(
    (item) => pathname === item.href || pathname.startsWith(`${item.href}/`),
  );
}
