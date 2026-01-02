import React from 'react';

export const EmbeddingOptions = ({ options, onChange }) => {
  return (
    <div className="grid grid-cols-2 gap-4 mb-4">
      <div>
        <label className="block text-sm font-medium text-gray-700">Embedding Provider</label>
        <select
          name="provider"
          value={options.provider}
          onChange={onChange}
          className="mt-1 block w-full rounded-md border-gray-300 shadow-sm"
        >
          <option value="huggingface">HuggingFace</option>
          <option value="openai">OpenAI</option>
        </select>
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700">Embedding Model</label>
        <input
          type="text"
          name="model"
          value={options.model}
          onChange={onChange}
          className="mt-1 block w-full rounded-md border-gray-300 shadow-sm"
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700">Index Path</label>
        <input
          type="text"
          name="indexPath"
          value={options.indexPath}
          onChange={onChange}
          className="mt-1 block w-full rounded-md border-gray-300 shadow-sm"
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700">Meta Path</label>
        <input
          type="text"
          name="metaPath"
          value={options.metaPath}
          onChange={onChange}
          className="mt-1 block w-full rounded-md border-gray-300 shadow-sm"
        />
      </div>
    </div>
  );
};

export const TextInput = ({ value, onChange, rows = 6, placeholder }) => {
  return (
    <textarea
      className="w-full p-2 border rounded-md mb-4"
      rows={rows}
      placeholder={placeholder}
      value={value}
      onChange={onChange}
    />
  );
};
