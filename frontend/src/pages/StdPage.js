import React, { useMemo, useState } from 'react';
import { AlertCircle } from 'lucide-react';
import { EmbeddingOptions, TextInput } from '../components/shared/ModelOptions';
import { postJson, getApiBaseUrl } from '../lib/api';

function normalizeLines(text) {
  return text
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter(Boolean);
}

const StdPage = () => {
  const [mode, setMode] = useState('single'); // single | batch
  const [input, setInput] = useState('A Priori Probability');
  const [topK, setTopK] = useState(5);

  const [embeddingOptions, setEmbeddingOptions] = useState({
    provider: 'huggingface',
    model: 'BAAI/bge-m3',
    indexPath: 'db/fin_terms_bge_m3.faiss',
    metaPath: 'db/fin_terms_meta.jsonl',
  });

  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleEmbeddingOptionChange = (e) => {
    const { name, value } = e.target;
    setEmbeddingOptions((prev) => ({ ...prev, [name]: value }));
  };

  const batchItems = useMemo(() => normalizeLines(input), [input]);

  const handleSubmit = async () => {
    setIsLoading(true);
    setError('');
    setResult(null);

    try {
      if (mode === 'single') {
        const data = await postJson('/api/fin/std', {
          text: input.trim(),
          topK: Number(topK) || 5,
          embeddingOptions,
        });
        setResult(data);
      } else {
        const data = await postJson('/api/fin/std/batch', {
          texts: batchItems,
          topK: Number(topK) || 5,
          embeddingOptions,
        });
        setResult(data);
      }
    } catch (e) {
      setError(e.message || String(e));
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="max-w-6xl mx-auto">
      <h1 className="text-3xl font-bold mb-2">金融术语标准化</h1>
      <p className="text-sm text-gray-600 mb-6">
        后端：<code className="bg-gray-200 px-1 rounded">{getApiBaseUrl()}</code>
      </p>

      <div className="grid grid-cols-3 gap-6">
        <div className="col-span-2 bg-white shadow-md rounded-lg p-6">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-xl font-semibold">输入</h2>
            <div className="flex items-center gap-2">
              <button
                className={`px-3 py-1 rounded border ${
                  mode === 'single' ? 'bg-gray-900 text-white' : 'bg-white'
                }`}
                onClick={() => setMode('single')}
              >
                单条
              </button>
              <button
                className={`px-3 py-1 rounded border ${
                  mode === 'batch' ? 'bg-gray-900 text-white' : 'bg-white'
                }`}
                onClick={() => setMode('batch')}
              >
                批量
              </button>
            </div>
          </div>

          <TextInput
            value={input}
            onChange={(e) => setInput(e.target.value)}
            rows={mode === 'single' ? 4 : 10}
            placeholder={mode === 'single' ? '请输入金融术语...' : '每行一个术语'}
          />

          <div className="grid grid-cols-2 gap-4 mb-4">
            <div>
              <label className="block text-sm font-medium text-gray-700">TopK</label>
              <input
                type="number"
                min={1}
                max={50}
                value={topK}
                onChange={(e) => setTopK(e.target.value)}
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm"
              />
            </div>
            {mode === 'batch' && (
              <div className="text-sm text-gray-600 flex items-end">
                共 {batchItems.length} 条
              </div>
            )}
          </div>

          <EmbeddingOptions options={embeddingOptions} onChange={handleEmbeddingOptionChange} />

          <button
            onClick={handleSubmit}
            className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 w-full"
            disabled={isLoading || (mode === 'single' ? !input.trim() : batchItems.length === 0)}
          >
            {isLoading ? '处理中...' : '标准化'}
          </button>
        </div>

        <div className="bg-white shadow-md rounded-lg p-6">
          <h2 className="text-xl font-semibold mb-2">说明</h2>
          <div className="text-sm text-gray-700 space-y-2">
            <p>本页面会调用后端接口获取 TopK 相似术语候选（FAISS + embedding）。</p>
            <p>
              默认索引路径：<code className="bg-gray-100 px-1 rounded">db/fin_terms_bge_m3.faiss</code>
            </p>
            <p>
              默认 meta 路径：<code className="bg-gray-100 px-1 rounded">db/fin_terms_meta.jsonl</code>
            </p>
            <p className="text-gray-500">提示：首次构建索引/首次加载模型可能比较慢。</p>
          </div>
        </div>
      </div>

      {(error || result) && (
        <div className="mt-6">
          {error && (
            <div className="bg-red-100 border-l-4 border-red-500 text-red-700 p-4 mb-6" role="alert">
              <p className="font-bold">错误：</p>
              <p>{error}</p>
            </div>
          )}
          {result && (
            <div className="bg-green-50 border border-green-200 text-green-900 p-4 rounded" role="alert">
              <p className="font-bold mb-2">结果：</p>
              <pre className="whitespace-pre-wrap break-words text-sm">{JSON.stringify(result, null, 2)}</pre>
            </div>
          )}
        </div>
      )}

      <div className="flex items-center text-yellow-700 bg-yellow-100 p-4 rounded-md mt-6">
        <AlertCircle className="mr-2" />
        <span>这是一个最小可用前端，用于辅助验证后端标准化能力；UI/功能可继续扩展。</span>
      </div>
    </div>
  );
};

export default StdPage;
