import { useState, useMemo } from "react";
import { X } from "lucide-react";
import { FeatureImportanceHeatmap } from "./FeatureImportanceHeatmap";

interface Props {
  classAttributions: number[][][][];
  classSampleCounts: number[];
  selectedClass: number | null;
  onSelectClass: (classIdx: number | null) => void;
}

export function ClassAttributionComparison({
  classAttributions,
  classSampleCounts,
  selectedClass,
  onSelectClass,
}: Props) {
  const classNames = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];

  const topFeaturesForSelectedClass = useMemo(() => {
    if (selectedClass === null || !classAttributions[selectedClass]) return [];

    const attribution = classAttributions[selectedClass][0];
    const features: Array<{ x: number; y: number; value: number }> = [];

    for (let y = 0; y < 28; y++) {
      for (let x = 0; x < 28; x++) {
        features.push({ x, y, value: attribution[y][x] });
      }
    }

    features.sort((a, b) => b.value - a.value);
    return features.slice(0, 5);
  }, [selectedClass, classAttributions]);

  if (!classAttributions || classAttributions.length === 0) {
    return (
      <div className="rounded-xl border border-gray-800 bg-[#111827] p-5">
        <div className="text-sm font-semibold text-gray-300 mb-4">类别级归因对比</div>
        <div className="h-[260px] flex items-center justify-center text-gray-500 text-sm">
          No data available
        </div>
      </div>
    );
  }

  if (selectedClass !== null) {
    const attribution = classAttributions[selectedClass];
    const sampleCount = classSampleCounts[selectedClass] || 0;

    return (
      <div className="rounded-xl border border-gray-800 bg-[#111827] p-5">
        <div className="text-sm font-semibold text-gray-300 mb-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span>类别 {classNames[selectedClass]} 详细归因</span>
            <span className="text-xs text-gray-500">({sampleCount} samples)</span>
          </div>
          <button
            onClick={() => onSelectClass(null)}
            className="p-1.5 rounded-lg hover:bg-gray-700 transition-colors text-gray-400 hover:text-gray-300"
          >
            <X size={16} />
          </button>
        </div>

        <div className="grid grid-cols-2 gap-6">
          <FeatureImportanceHeatmap
            attribution={attribution}
            title={`Class ${classNames[selectedClass]} Heatmap`}
            showLegend={true}
            highlightClass={selectedClass}
          />

          <div className="rounded-lg border border-gray-700 bg-[#0a0e1a] p-4">
            <div className="text-xs font-semibold text-gray-400 mb-3">Top-5 贡献特征坐标</div>
            <div className="space-y-2">
              {topFeaturesForSelectedClass.map((feature, idx) => (
                <div
                  key={idx}
                  className="flex items-center justify-between p-2 rounded-lg bg-[#111827] border border-gray-700"
                >
                  <div className="flex items-center gap-3">
                    <span className="w-6 h-6 rounded-full bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center text-xs font-bold text-white">
                      {idx + 1}
                    </span>
                    <span className="text-sm text-gray-300 font-mono">
                      ({feature.x}, {feature.y})
                    </span>
                  </div>
                  <span className="text-sm font-semibold text-cyan-400">
                    {feature.value.toFixed(2)}%
                  </span>
                </div>
              ))}
            </div>

            <div className="mt-4 pt-4 border-t border-gray-700">
              <div className="text-xs text-gray-500 mb-2">颜色图例</div>
              <div className="flex items-center gap-2">
                <div
                  style={{
                    width: "100%",
                    height: "16px",
                    background: "linear-gradient(to right, #3b82f6, #ef4444)",
                    borderRadius: "4px",
                  }}
                />
              </div>
              <div className="flex justify-between text-xs text-gray-500 mt-1">
                <span>低贡献 (0%)</span>
                <span>高贡献 (100%)</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-gray-800 bg-[#111827] p-5">
      <div className="text-sm font-semibold text-gray-300 mb-4">类别级归因对比</div>
      <p className="text-xs text-gray-500 mb-4">点击任意缩略图查看该类别的详细归因分析</p>

      <div className="grid grid-cols-5 gap-4">
        {classNames.map((className, idx) => {
          const attribution = classAttributions[idx];
          const sampleCount = classSampleCounts[idx] || 0;

          return (
            <div
              key={idx}
              className="group"
            >
              <div
                className="relative cursor-pointer transition-transform hover:scale-105"
                onClick={() => onSelectClass(idx)}
              >
                <FeatureImportanceHeatmap
                  attribution={attribution}
                  thumbnail={true}
                  showLegend={false}
                />
                <div className="absolute inset-0 flex items-center justify-center bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity rounded-xl">
                  <span className="text-xs text-white font-semibold">查看详情</span>
                </div>
              </div>
              <div className="text-center mt-2">
                <div className="text-sm font-semibold text-gray-300">Class {className}</div>
                <div className="text-xs text-gray-500">{sampleCount} samples</div>
              </div>
            </div>
          );
        })}
      </div>

      <div className="mt-4 pt-4 border-t border-gray-700 flex items-center justify-center gap-6">
        <div className="flex items-center gap-2">
          <div
            style={{
              width: "40px",
              height: "12px",
              background: "linear-gradient(to right, #3b82f6, #ef4444)",
              borderRadius: "2px",
            }}
          />
          <span className="text-xs text-gray-500">蓝: 低贡献 → 红: 高贡献</span>
        </div>
      </div>
    </div>
  );
}
