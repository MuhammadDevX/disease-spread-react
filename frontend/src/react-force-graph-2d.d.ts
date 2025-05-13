declare module 'react-force-graph-2d' {
  import * as React from 'react';
  interface ForceGraph2DProps {
    graphData: any;
    nodeLabel?: (node: any) => string;
    nodeAutoColorBy?: any;
    nodeCanvasObject?: (node: any, ctx: CanvasRenderingContext2D, globalScale: number) => void;
    linkColor?: (link: any) => string;
    linkWidth?: (link: any) => number;
    width?: number;
    height?: number;
    backgroundColor?: string;
    enableNodeDrag?: boolean;
    cooldownTicks?: number;
    onEngineStop?: () => void;
  }
  const ForceGraph2D: React.FC<ForceGraph2DProps>;
  export default ForceGraph2D;
} 