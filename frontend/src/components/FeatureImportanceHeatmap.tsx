import { useState, useMemo } from "react";

interface Props {
  attribution: number[][][];
  title?: string;
  showLegend?: boolean;
  thumbnail?: boolean;
  onClick?: () => void;
  highlightClass?: number | null;
}

export function FeatureImportanceHeatmap({
  attribution,
  title = "Feature Importance",
  showLegend = true,
  thumbnail = false,
  onClick,
  highlightClass,
}: Props) {
  const [hoveredCell, setHoveredCell] = useState<{ x: number; y: number; value: number } | null>(null);

  const grid = useMemo(() => {
    if (!attribution || !attribution[0]) return [];
    return attribution[0];
  }, [attribution]);

  const size = 28;
  const cellSize = thumbnail ? 8 : 24;

  const getColor = (value: number) => {
    const v = Math.max(0, Math.min(100, value));
    const t = v / 100;
    
    const r = Math.round(59 * (1 - t) + 239 * t);
    const g = Math.round(130 * (1 - t) + 68 * t);
    const b = Math.round(246 * (1 - t) + 68 * t);
    
    return `rgb(${r}, ${g}, ${b})`;
  };

  const handleMouseEnter = (x: number, y: number, value: number) => {
    if (!thumbnail) {
      setHoveredCell({ x, y, value });
    }
  };

  const handleMouseLeave = () => {
    setHoveredCell(null);
  };

  const tooltipStyle: React.CSSProperties = hoveredCell
    ? {
        position: "absolute" as const,
        left: `${Math.min(hoveredCell.x * cellSize + 30, size * cellSize - 150)}px`,
        top: `${Math.max(hoveredCell.y * cellSize - 40, 0)}px`,
        background: "rgba(17, 24, 39, 0.95)",
        border: "1px solid #374151",
        borderRadius: "6px",
        padding: "6px 10px",
        fontSize: "11px",
        color: "#e5e7eb",
        pointerEvents: "none" as const,
        zIndex: 10,
        whiteSpace: "nowrap" as const,
      }
    : {};

  if (grid.length === 0) {
    return (
      <div className="flex items-center justify-center text-gray-500 text-sm" style={{ height: size * cellSize }}>
        No data available
      </div>
    );
  }

  return (
    <div
      className={`rounded-xl border ${thumbnail ? "border-gray-700" : "border-gray-800"} bg-[#111827] ${thumbnail ? "p-1" : "p-5"} relative ${onClick ? "cursor-pointer hover:ring-2 hover:ring-cyan-500/50 transition-all" : ""}`}
      onClick={onClick}
    >
      {!thumbnail && (
        <div className="text-sm font-semibold text-gray-300 mb-4 flex items-center justify-between">
          <span>{title}</span>
          {highlightClass !== null && highlightClass !== undefined && (
            <span className="text-xs text-cyan-400">Class {highlightClass}</span>
          )}
        </div>
      )}

      <div className="flex gap-4">
        <div
          className="relative"
          style={{
            display: "grid",
            gridTemplateColumns: `repeat(${size}, ${cellSize}px)`,
            gridTemplateRows: `repeat(${size}, ${cellSize}px)`,
            gap: "1px",
            background: "#1f2937",
            padding: "1px",
            borderRadius: "4px",
          }}
          onMouseLeave={handleMouseLeave}
        >
          {grid.map((row, y) =>
            row.map((value, x) => (
              <div
                key={`${x}-${y}`}
                style={{
                  width: cellSize,
                  height: cellSize,
                  backgroundColor: getColor(value),
                }}
                onMouseEnter={() => handleMouseEnter(x, y, value)}
              />
            ))
          )}

          {hoveredCell && (
            <div style={tooltipStyle}>
              像素坐标({hoveredCell.x},{hoveredCell.y}): 贡献值{hoveredCell.value.toFixed(2)}%
            </div>
          )}
        </div>

        {showLegend && !thumbnail && (
          <div className="flex flex-col items-center">
            <div className="text-xs text-gray-400 mb-2">贡献度</div>
            <div
              style={{
                width: "20px",
                height: size * cellSize,
                background: "linear-gradient(to top, #3b82f6, #ef4444)",
                borderRadius: "4px",
              }}
            />
            <div className="text-xs text-gray-400 mt-2">高</div>
            <div className="text-xs text-cyan-400 font-mono mt-1">100%</div>
            <div className="flex-1" />
            <div className="text-xs text-cyan-400 font-mono mb-1">0%</div>
            <div className="text-xs text-gray-400">低</div>
          </div>
        )}
      </div>
    </div>
  );
}
