"use client";

import React, { useState, useRef, useEffect } from 'react';
import { Send, Paperclip, FileText, CheckCircle, Clock, Trash2, Filter, ChevronRight, MessageSquare, Download, Trash } from 'lucide-react';

interface Document {
  filename: string;
  uploaded_at: string;
  status: string;
  type: string;
}

export default function ChatPage() {
  const [messages, setMessages] = useState<any[]>([
    {
      id: 1,
      type: 'bot',
      text: '반갑습니다, 팀장님! Admate AI 어시스턴트입니다. 어떤 서류의 정보를 찾아드릴까요?',
      sources: []
    }
  ]);
  const [inputText, setInputText] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [uploadedDocs, setUploadedDocs] = useState<Document[]>([]);
  const [selectedFilter, setSelectedFilter] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  // 스크롤 자동 이동
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  // 문서 목록 가져오기 함수 (폴링 및 상태 갱신용)
  const fetchDocs = async () => {
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';
    try {
      const response = await fetch(`${apiUrl}/api/v1/docs/list`);
      if (response.ok) {
        const data = await response.json();
        setUploadedDocs(data.documents || []);
        return data.documents || [];
      }
    } catch (error) {
      console.error('문서 목록 로드 실패:', error);
    }
    return [];
  };

  // 초기 로드 시 문서 목록 가져오기 및 폴링 설정
  useEffect(() => {
    fetchDocs();

    // 5초마다 상태 체크 (이미지 분석 중인 문서가 있을 경우 대비)
    const interval = setInterval(() => {
      fetchDocs();
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  // 채팅 전송 핸들러 (스트리밍 지원)
  const handleSend = async () => {
    if (!inputText.trim() || isStreaming) return;

    const userMsgText = inputText;
    const userMessage = { id: Date.now(), type: 'user', text: userMsgText, sources: [] };
    setMessages(prev => [...prev, userMessage]);
    setInputText('');
    setIsStreaming(true);

    // 봇 응답용 빈 메시지 생성
    const botMsgId = Date.now() + 1;
    setMessages(prev => [...prev, { id: botMsgId, type: 'bot', text: '', sources: [] }]);

    const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';
    try {
      const response = await fetch(`${apiUrl}/api/v1/chat/ask/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: userMsgText,
          filter_source: selectedFilter
        }),
      });

      if (!response.body) throw new Error('No body');

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let done = false;
      let accumulatedText = '';
      let sources: any[] = [];

      while (!done) {
        const { value, done: doneReading } = await reader.read();
        done = doneReading;
        const chunk = decoder.decode(value, { stream: true });

        // 청크 처리 (소스 정보 vs 텍스트 정보)
        const lines = chunk.split('\n');
        for (const line of lines) {
          if (line.startsWith('__SOURCES__:')) {
            try {
              sources = JSON.parse(line.replace('__SOURCES__:', ''));
            } catch (e) {
              console.error('Source parsing error', e);
            }
          } else if (line.trim() || chunk.length < 5) { // 빈 줄 방지 및 미세 청크 허용
            accumulatedText += line;
          }
        }

        // 실시간 메시지 업데이트
        setMessages(prev => prev.map(m =>
          m.id === botMsgId ? { ...m, text: accumulatedText, sources: sources } : m
        ));
      }
    } catch (error) {
      setMessages(prev => prev.map(m =>
        m.id === botMsgId ? { ...m, text: '답변을 생성하는 동안 오류가 발생했습니다.' } : m
      ));
    } finally {
      setIsStreaming(false);
    }
  };

  // 문서 삭제 핸들러
  const handleDeleteDoc = async (filename: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!confirm(`'${filename}' 문서를 삭제하시겠습니까?`)) return;

    const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';
    try {
      const response = await fetch(`${apiUrl}/api/v1/docs/delete/${encodeURIComponent(filename)}`, {
        method: 'DELETE'
      });
      if (response.ok) {
        setUploadedDocs(prev => prev.filter(d => d.filename !== filename));
        if (selectedFilter === filename) setSelectedFilter(null);
      }
    } catch (error) {
      alert('삭제 중 오류가 발생했습니다.');
    }
  };

  // 파일 업로드 통합 처리
  const processUpload = async (files: FileList) => {
    setIsUploading(true);
    const formData = new FormData();
    Array.from(files).forEach(file => formData.append('files', file));

    const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';
    try {
      const response = await fetch(`${apiUrl}/api/v1/docs/upload`, {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        await fetchDocs(); // 즉시 목록 갱신 (텍스트 인덱싱 결과 확인)
        alert('문서의 텍스트 학습이 완료되었습니다! 이미지는 백그라운드에서 분석 중입니다.');
      }
    } catch (error) {
      alert('업로드 중 오류가 발생했습니다.');
    } finally {
      setIsUploading(false);
      setIsDragging(false);
    }
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files) processUpload(event.target.files);
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files) processUpload(e.dataTransfer.files);
  };

  return (
    <div
      className="flex h-screen bg-[#F9FAFB] font-sans text-[#111827] relative"
      onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={onDrop}
    >
      {isDragging && (
        <div className="absolute inset-0 bg-blue-500/10 border-4 border-dashed border-blue-500 z-50 flex items-center justify-center backdrop-blur-sm transition-all">
          <div className="bg-white p-8 rounded-3xl shadow-2xl flex flex-col items-center gap-4">
            <Download size={64} className="text-blue-500 animate-bounce" />
            <p className="text-2xl font-black text-blue-600">이곳에 문서를 놓아주세요</p>
          </div>
        </div>
      )}

      <input type="file" accept=".pdf,.xlsx,.xls,.csv,.txt" multiple hidden ref={fileInputRef} onChange={handleFileUpload} />

      {/* Sidebar */}
      <aside className="w-80 bg-white border-r border-gray-100 flex flex-col shadow-xl z-10 transition-all overflow-hidden">
        <div className="p-6 border-b border-gray-50 bg-gradient-to-br from-white to-blue-50/30">
          <h1 className="text-2xl font-black tracking-tight text-[#1D4ED8] flex items-center gap-2">
            Admate <span className="text-blue-400">AI</span>
          </h1>
        </div>

        <nav className="flex-1 overflow-y-auto custom-scrollbar p-4 space-y-6">
          <div>
            <div className="flex items-center justify-between mb-4 px-2">
              <p className="text-[10px] font-black text-gray-400 uppercase tracking-[0.2em]">지식 베이스</p>
              <button
                onClick={() => setSelectedFilter(null)}
                className={`text-[10px] font-bold px-2 py-1 rounded-md transition-all ${!selectedFilter ? 'bg-blue-600 text-white shadow-md' : 'text-gray-400 hover:text-gray-600'}`}
              >
                전체보기
              </button>
            </div>

            <div className="space-y-2">
              {uploadedDocs.length > 0 ? (
                uploadedDocs.map((doc, idx) => (
                  <div
                    key={idx}
                    onClick={() => setSelectedFilter(doc.filename)}
                    className={`group relative p-3 rounded-2xl border transition-all cursor-pointer hover:shadow-md ${selectedFilter === doc.filename ? 'bg-blue-50 border-blue-200 shadow-sm' : 'bg-white border-gray-100'}`}
                  >
                    <div className="flex items-center gap-3">
                      <div className={`p-2 rounded-xl ${selectedFilter === doc.filename ? 'bg-blue-600 text-white' : 'bg-gray-50 text-gray-400 group-hover:bg-blue-100 group-hover:text-blue-600'}`}>
                        <FileText size={18} />
                      </div>
                      <div className="flex-1 overflow-hidden">
                        <p className={`text-xs font-bold truncate ${selectedFilter === doc.filename ? 'text-blue-700' : 'text-gray-700'}`} title={doc.filename}>{doc.filename}</p>
                        <div className="flex items-center gap-2 mt-1">
                          {doc.status === 'indexing_images' ? (
                            <span className="text-[9px] text-blue-500 font-black animate-pulse flex items-center gap-0.5">
                              <Clock size={8} /> 이미지 분석 중...
                            </span>
                          ) : (
                            <span className="text-[9px] text-emerald-500 font-black flex items-center gap-0.5" title="모든 내용 학습 완료">
                              <CheckCircle size={8} /> {doc.status || 'Indexed'}
                            </span>
                          )}
                          {doc.uploaded_at && (
                            <span className="text-[9px] text-gray-400 flex items-center gap-0.5">
                              <Clock size={8} /> {new Date(doc.uploaded_at).toLocaleDateString()}
                            </span>
                          )}
                        </div>
                      </div>
                      <button
                        onClick={(e) => handleDeleteDoc(doc.filename, e)}
                        className="opacity-0 group-hover:opacity-100 p-1.5 text-gray-400 hover:text-red-500 hover:bg-red-50 rounded-lg transition-all"
                      >
                        <Trash size={14} />
                      </button>
                    </div>
                  </div>
                ))
              ) : (
                <div className="p-8 border-2 border-dashed border-gray-100 rounded-3xl text-center">
                  <p className="text-[11px] text-gray-400 font-medium">문서를 업로드하여<br />지식을 학습시켜주세요</p>
                </div>
              )}
            </div>
          </div>

          <div className="pt-4 border-t border-gray-50">
            <p className="text-[10px] font-black text-gray-400 uppercase tracking-[0.2em] mb-4 px-2 text-center">퀵 프롬프트 (FAQ)</p>
            <div className="grid grid-cols-1 gap-2">
              {['광고 수수료 정책 알려줘', '단가표 일정 확인해줘', '업로드된 문서 요약해줘'].map((q, i) => (
                <button
                  key={i}
                  onClick={() => setInputText(q)}
                  className="text-[11px] text-left p-3 rounded-xl bg-gray-50 border border-gray-100 text-gray-600 hover:bg-white hover:border-blue-200 hover:text-blue-600 transition-all font-medium"
                >
                  " {q} "
                </button>
              ))}
            </div>
          </div>
        </nav>

        <div className="p-4 border-t border-gray-50 bg-gray-50/30">
          <button
            disabled={isUploading}
            onClick={() => fileInputRef.current?.click()}
            className="w-full py-4 bg-[#1D4ED8] text-white rounded-2xl flex items-center justify-center gap-2 font-black shadow-xl shadow-blue-200 hover:bg-blue-700 hover:scale-[1.02] active:scale-95 transition-all disabled:opacity-50"
          >
            <Paperclip size={20} /> {isUploading ? '학습 중...' : '신규 문서 학습'}
          </button>
        </div>
      </aside>

      {/* Main Area */}
      <main className="flex-1 flex flex-col bg-white overflow-hidden">
        <header className="h-16 border-b border-gray-50 flex items-center justify-between px-8 bg-white/80 backdrop-blur-md z-10">
          <div className="flex items-center gap-3">
            <div className={`w-2.5 h-2.5 rounded-full ${isStreaming ? 'bg-blue-500 animate-ping' : 'bg-emerald-500 shadow-sm shadow-emerald-200'}`}></div>
            <span className="font-black text-sm text-gray-700 tracking-tight">
              {selectedFilter ? `[필터링됨] ${selectedFilter}` : 'Admate Chat Hub'}
            </span>
          </div>
          <div className="flex items-center gap-4">
            <div className="text-[11px] font-bold text-gray-400 bg-gray-50 px-3 py-1.5 rounded-full border border-gray-100">
              GPT-4o Multimodal Engine
            </div>
          </div>
        </header>

        <section ref={scrollRef} className="flex-1 overflow-y-auto p-12 space-y-8 bg-[url('https://www.transparenttextures.com/patterns/cubes.png')] bg-fixed">
          {messages.map((msg) => (
            <div key={msg.id} className={`flex ${msg.type === 'user' ? 'justify-end' : 'justify-start'} animate-in fade-in slide-in-from-bottom-2 duration-300`}>
              <div className={`relative max-w-[80%] p-6 rounded-3xl shadow-lg transition-all ${msg.type === 'user' ? 'bg-[#1D4ED8] text-white shadow-blue-100 ring-4 ring-blue-50/20' : 'bg-white border border-gray-100 text-gray-800'}`}>
                {msg.type === 'bot' && (
                  <div className="absolute -left-10 top-2 w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center text-blue-600 font-black text-xs shadow-inner">AI</div>
                )}
                <p className="text-[13px] leading-[1.8] whitespace-pre-wrap font-medium">{msg.text}</p>
                {msg.sources && msg.sources.length > 0 && (
                  <div className={`mt-4 pt-4 border-t ${msg.type === 'user' ? 'border-blue-400/30' : 'border-gray-50'}`}>
                    <div className="flex items-center justify-between mb-2">
                      <p className={`text-[9px] font-black uppercase tracking-widest ${msg.type === 'user' ? 'text-blue-100' : 'text-gray-400'}`}>Verified Sources</p>
                      <span className={`text-[9px] font-bold px-2 py-0.5 rounded-full ${msg.type === 'user' ? 'bg-blue-400/30' : 'bg-blue-50 text-blue-500'}`}>
                        신뢰도 {Math.round((msg.sources[0]?.score || 0.8) * 100)}%
                      </span>
                    </div>
                    <div className="flex flex-wrap gap-1.5">
                      {msg.sources.map((src: any, i: number) => (
                        <div key={i} className={`flex items-center gap-1.5 text-[10px] px-2.5 py-1 rounded-lg border transition-all ${msg.type === 'user' ? 'bg-white/10 border-white/20 text-blue-50' : 'bg-gray-50 border-gray-100 text-[#3B82F6]'}`}>
                          <FileText size={10} />
                          <span className="font-bold">{src.name}</span>
                          {src.page && <span className="opacity-60">p.{src.page}</span>}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          ))}
          {isStreaming && (
            <div className="flex justify-start animate-pulse">
              <div className="bg-gray-50 border border-gray-100 p-4 rounded-3xl text-gray-400 text-xs font-bold">
                생각하며 답변을 작성 중입니다...
              </div>
            </div>
          )}
        </section>

        <footer className="p-8 bg-white border-t border-gray-50 z-10">
          <div className="max-w-4xl mx-auto relative group">
            <div className="absolute -inset-1 bg-gradient-to-r from-blue-600 to-indigo-600 rounded-3xl blur opacity-10 group-focus-within:opacity-20 transition duration-1000"></div>
            <div className="relative flex gap-3 p-2 bg-white border border-gray-200 rounded-3xl shadow-xl shadow-blue-50/50">
              <button
                onClick={() => fileInputRef.current?.click()}
                className="p-4 text-gray-400 hover:text-blue-600 hover:bg-blue-50 rounded-2xl transition-all"
              >
                <Paperclip size={24} />
              </button>
              <input
                type="text"
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleSend()}
                placeholder={selectedFilter ? `'${selectedFilter}' 내에서 답변을 찾아드릴까요?` : "궁금한 지식을 물어보세요..."}
                className="flex-1 p-4 bg-transparent focus:outline-none text-[14px] font-semibold text-gray-700 placeholder:text-gray-300"
              />
              <button
                onClick={handleSend}
                disabled={isStreaming}
                className={`p-4 rounded-2xl transition-all shadow-lg ${isStreaming ? 'bg-gray-200 text-gray-400 cursor-not-allowed' : 'bg-[#1D4ED8] text-white hover:bg-blue-700 active:scale-95 shadow-blue-200 active:shadow-none'}`}
              >
                <Send size={24} />
              </button>
            </div>
            <p className="mt-3 text-[10px] text-center text-gray-400 font-bold">Admate AI는 문서의 내용을 기반으로 가장 정확한 정보를 실시간으로 추출합니다.</p>
          </div>
        </footer>
      </main>

      <style jsx global>{`
        .custom-scrollbar::-webkit-scrollbar {
          width: 4px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: transparent;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: #E5E7EB;
          border-radius: 10px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: #D1D5DB;
        }
      `}</style>
    </div>
  );
}
