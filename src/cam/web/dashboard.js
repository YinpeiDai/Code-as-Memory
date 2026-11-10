/* Directory-backed workspace. Gallery links open a page; episode navigation stays there. */
(() => {
  'use strict';
  const $ = s => document.querySelector(s), main = $('#main');
  const names = {robomme:'RoboMME', mikasa_robo:'MIKASA-Robo', mikasa_robo_vla_suite:'MIKASA-Robo VLA suite (5 tasks)', rmbench:'RMBench', robomemarena:'RoboMemArena'};
  const initials = {robomme:'MM', mikasa_robo:'MK', rmbench:'RM', robomemarena:'MA'};
  const labels = {demo_only_25:'25 demonstrations', demo_only_50:'Up to 50 demonstrations', demo_interaction_50:'50 demonstrations + interaction', demo_only_25_observed_motion:'25 demos · current interface', demo_only_50_observed_motion:'50 demos · current interface', pool:'Full demonstration pool'};
  const esc = v => String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const fmt = n => Number(n || 0).toLocaleString(), pct = n => n == null ? '—' : (100*n).toFixed(2)+'%';
  const pretty = s => labels[s] || String(s || '').replaceAll('_',' ');
  let data, allData, page, params, request = 0;
  function route(){const [p,q] = location.hash.slice(1).split('?');page=p || 'overview';params=new URLSearchParams(q || '');}
  function url(p, values={}){const q=new URLSearchParams(Object.entries({history:params?.get('history'),...values}).filter(([,v])=>v!=null && v!==''));return '#'+p+(q.size?'?'+q:'');}
  const selection = () => Object.fromEntries(params.entries());
  const badge = (text, kind='') => `<span class="badge ${kind}">${esc(text)}</span>`;
  const hero = (eyebrow,title,description,actions='') => `<div class="hero"><div><span class="eyebrow">${eyebrow}</span><h1>${title}</h1><p>${description}</p></div>${actions?`<div class="hero-actions">${actions}</div>`:''}</div>`;
  const blank = text => `<div class="empty">${esc(text)}</div>`;
  const runById = id => data.runs.find(r=>r.id===id);
  const demoById = id => data.demos.find(d=>d.id===id);
  const normalized = setting => String(setting||'').replace(/_observed_motion$/, '');
  function filtered(rows){
    const requested=params.get('setting'),best=data.default_settings||{robomme:'demo_only_25',mikasa_robo:'demo_interaction_50',rmbench:'demo_only_50',robomemarena:'demo_interaction_50'};
    return rows.filter(r=>{
      if(params.get('benchmark')&&r.benchmark!==params.get('benchmark'))return false;
      if(requested&&requested!=='__all__'&&r.setting!==requested)return false;
      if(!requested&&!params.get('run')&&page!=='episode'){
        const desired=r.kind==='demo'?'pool':best[r.benchmark];
        if(rows.some(x=>x.benchmark===r.benchmark&&normalized(x.setting)===desired)&&normalized(r.setting)!==desired)return false;
      }
      return (!params.get('task')||r.task===params.get('task'))&&(!params.get('run')||r.run_id===params.get('run')||r.id===params.get('run'))&&(!params.get('q')||`${r.task} ${pretty(r.setting)} ${r.episode ?? ''}`.toLowerCase().includes(params.get('q').toLowerCase()));
    });
  }
  function applyView(){
    const history=params.get('history')==='1'||['run','episode'].includes(page)||params.has('run');
    const runs=allData.runs.filter(r=>history||r.latest!==false),ids=new Set(runs.map(r=>r.id));
    const tags=new Set(runs.map(r=>r.group_tag||r.tag));
    data={...allData,batches:allData.batches.filter(b=>history||b.status==='running'||tags.has(b.tag)),runs,episodes:allData.episodes.filter(e=>ids.has(e.run_id)),demos:allData.demos.filter(d=>!d.run_id||ids.has(d.run_id))};
    data.groups=allData.groups.map(g=>{
      const tasks=g.tasks.filter(id=>ids.has(id)),selected=runs.filter(r=>tasks.includes(r.id)),summary={};
      for(const key of ['expected','recorded','scored','successes','infrastructure_errors'])summary[key]=selected.reduce((n,r)=>n+r.summary[key],0);
      summary.complete=selected.length>0&&selected.every(r=>r.summary.complete);
      summary.success_rate=summary.complete&&summary.expected?summary.successes/summary.expected:null;
      return {...g,tasks,summary};
    }).filter(g=>g.tasks.length);
    $('#view-options').hidden=['run','episode'].includes(page);
    $('#view-label').textContent=(params.get('setting')==='__all__'?'All settings':params.get('setting')?pretty(params.get('setting')):'Best published settings')+' · '+(history?'all runs':'latest runs');
    $('#settings-toggle').textContent=params.get('setting')==='__all__'?'Use best settings':'Show all settings';
    $('#settings-toggle').onclick=()=>{location.hash=url(page,{...selection(),setting:params.get('setting')==='__all__'?null:'__all__',p:1});};
    $('#history-toggle').hidden=data.archive;
    $('#history-toggle').textContent=history?'Use latest runs':'Show history';
    $('#history-toggle').onclick=()=>{location.hash=url(page,{...selection(),history:history?null:'1',p:1});};
  }
  function filters(rows, outcome=false){
    const benches=[...new Set(rows.map(r=>r.benchmark))], scoped=rows.filter(r=>!params.get('benchmark')||r.benchmark===params.get('benchmark'));
    const settings=[...new Set(scoped.map(r=>r.setting))].sort(),tasks=[...new Set(scoped.map(r=>r.task))].sort();
    const select=(key,title,values,label=x=>x)=>`<label>${title}<select data-filter="${key}"><option value="">${key==='setting'?'Best published setting':'All '+title.toLowerCase()}</option>${key==='setting'?`<option value="__all__"${params.get(key)==='__all__'?' selected':''}>All settings</option>`:''}${values.map(v=>`<option value="${esc(v)}"${params.get(key)===v?' selected':''}>${esc(label(v))}</option>`).join('')}</select></label>`;
    return `<div class="filters">${select('benchmark','Benchmarks',benches,x=>names[x])}${select('setting','Settings',settings,pretty)}${select('task','Tasks',tasks)}${outcome?select('outcome','Outcomes',['success','failure','error'],x=>({success:'Task success',failure:'Task failure',error:'Infrastructure error'}[x])):''}<label class="search">Search<input type="search" id="search" placeholder="Find a task or episode…" value="${esc(params.get('q'))}"></label></div>`;
  }
  function bindFilters(){
    document.querySelectorAll('[data-filter]').forEach(el=>el.onchange=()=>{const values=selection();values[el.dataset.filter]=el.value;values.p=1;if(el.dataset.filter==='benchmark'){delete values.setting;delete values.task;} location.hash=url(page,values);});
    const search=$('#search');if(search){let timer;search.oninput=()=>{clearTimeout(timer);const start=search.selectionStart;timer=setTimeout(()=>{history.replaceState(null,'',url(page,{...selection(),q:search.value,p:1}));render();const input=$('#search');input?.focus();input?.setSelectionRange(start,start);},220);};}
  }
  function overview(){
    const viewRuns=filtered(data.runs);
    const active=viewRuns.filter(r=>['running','starting'].includes(r.development_status));
    const evaluating=viewRuns.filter(r=>r.evaluation_status==='running');
    const frozen=viewRuns.filter(r=>r.frozen),scored=viewRuns.reduce((n,r)=>n+r.summary.scored,0),expected=viewRuns.reduce((n,r)=>n+r.expected,0);
    main.innerHTML=hero('THE EXPERIMENT WORKSPACE','Experience becomes memory.','Follow robot memory development, inspect the evidence, and compare frozen policies across four benchmarks.',`<a class="button" href="#results">Explore results <span>↗</span></a><a class="button secondary" href="#development">View development</a>`)+
      `<div class="metrics">${[[active.length,'Developing now',`${viewRuns.length} task runs indexed`],[frozen.length,'Frozen policies','Ready for evaluation'],[scored,'Scored episodes',`${fmt(expected)} expected across indexed runs`],[filtered(data.demos).length,'Demonstrations','Browse recorded training examples']].map(([n,t,s])=>`<div class="metric"><span>${t}</span><strong>${fmt(n)}</strong><small>${esc(s)}</small></div>`).join('')}</div>`+
      `${(data.batches||[]).length?`<div class="section-head"><div><h2>Live queues</h2><p>Includes work waiting for preparation and GPU admission.</p></div></div><div class="result-grid">${data.batches.map(b=>`<article class="result-card"><div class="row"><h3>${esc(b.kind)}</h3>${badge(b.status,b.status==='running'?'pending':'')}</div><p class="detail-meta">${esc(b.tag)}</p><strong>${fmt(b.completed)} / ${fmt(b.expected)} ${esc(b.unit)}</strong><div class="bar"><i style="width:${b.expected?Math.min(100,100*b.completed/b.expected):0}%"></i></div><p>${fmt(b.running)} running · all requested settings</p></article>`).join('')}</div>`:''}<div class="section-head"><div><h2>Across the benchmarks</h2><p>Development and evaluation remain linked to their original runs.</p></div><a class="text-link" href="#evaluation">Open evaluation gallery ↗</a></div><div class="benchmark-grid">${Object.keys(names).map(b=>{const runs=viewRuns.filter(r=>r.benchmark===b),n=runs.reduce((s,r)=>s+r.summary.scored,0),total=runs.reduce((s,r)=>s+r.expected,0),done=runs.filter(r=>r.frozen).length;return `<a class="benchmark-card" href="${url('results',{benchmark:b})}"><div class="benchmark-top"><span class="benchmark-icon">${initials[b]}</span>${badge(n===total&&total?'Complete':runs.length?'In progress':'No runs',n===total&&total?'':'pending')}</div><h3>${names[b]}</h3><p>${new Set(runs.map(r=>r.task)).size} tasks · ${new Set(runs.map(r=>r.setting)).size} settings</p><div style="margin-top:23px" class="row"><span>Frozen policies</span><b>${done}/${runs.length}</b></div><div class="bar"><i style="width:${runs.length?100*done/runs.length:0}%"></i></div><div class="row muted"><span>Evaluation</span><span>${fmt(n)}/${fmt(total)} scored</span></div></a>`;}).join('')}</div>`+
      `<div class="section-head"><div><h2>${active.length?'Live development':evaluating.length?'Live evaluation':'Run activity'}</h2><p>Refreshes automatically. Opening an episode keeps playback uninterrupted.</p></div><a class="text-link" href="${data.downloads.progress}" target="_blank" rel="noopener">Progress file ↗</a></div><div class="panel activity">${(active.length?active:evaluating.length?evaluating:viewRuns.slice(-6)).map(r=>`<a href="${url('run',{id:r.id})}" target="_blank" rel="noopener"><span><b>${esc(r.task)}</b><br><span style="color:var(--muted)">${names[r.benchmark]} · ${esc(pretty(r.setting))}</span></span><span>${badge(r.evaluation_status==='running'?'evaluating':r.development_status)} &nbsp; ${r.summary.scored}/${r.expected} scored</span></a>`).join('')||'No task runs have been prepared yet.'}</div>`;
  }
  function results(){
    const runs=filtered(data.runs),groups=data.groups.filter(g=>runs.some(r=>g.tasks.includes(r.id)));
    main.innerHTML=hero('FROZEN POLICY RESULTS','Evidence, at every scale.','Compare complete evaluations, then open a task to inspect its episodes. Partial coverage and sanity checks are labelled separately.',`<a class="button secondary" href="${data.downloads.csv}" download>Export CSV ↓</a>`)+filters(data.runs)+
      `<div class="result-grid">${groups.map(g=>{const s=g.summary;return `<article class="result-card"><div class="row"><span class="eyebrow" style="margin:0">${names[g.benchmark]}</span>${badge(g.scope==='sanity'?'Sanity check':s.complete?'Complete':'Partial',s.complete?'':'pending')}</div><h3>${esc(pretty(g.setting))}</h3><strong class="rate">${pct(s.success_rate)}</strong><p>${fmt(s.successes)} successful · ${fmt(s.scored)}/${fmt(s.expected)} scored · ${g.tasks.length} task${g.tasks.length===1?'':'s'}</p><div class="bar"><i style="width:${s.expected?Math.min(100,100*s.scored/s.expected):0}%"></i></div><small>${g.scope==='sanity'?'Selected episodes for execution checks; not the full benchmark.':esc(g.tag==='published'?'Published experiment':g.tag)}</small><a class="text-link" href="${url('evaluation',{benchmark:g.benchmark,setting:g.setting})}">Inspect evaluation episodes ↗</a></article>`;}).join('')}</div>`+
      `<div class="section-head"><div><h2>Task-level results</h2><p>Success requires the benchmark’s official task outcome. Infrastructure errors remain separate.</p></div><span class="gallery-meta">${runs.length} task runs</span></div>`+
      `<div class="table-wrap"><table><thead><tr><th>Task</th><th>Benchmark / setting</th><th>Successes</th><th>Scored</th><th>Success rate</th><th>Evidence</th></tr></thead><tbody>${runs.map(r=>`<tr><td><a href="${url('run',{id:r.id})}" target="_blank" rel="noopener">${esc(r.task)} ↗</a></td><td>${names[r.benchmark]}<br><small>${esc(pretty(r.setting))}${r.scope==='sanity'?' · sanity':''}</small></td><td>${r.summary.successes}/${r.expected}</td><td>${r.summary.scored}/${r.expected}</td><td>${pct(r.summary.success_rate)}</td><td><a class="text-link" href="${url('evaluation',{run:r.id})}">${r.media_count?'Videos + results':'Results'} →</a></td></tr>`).join('')}</tbody></table></div>`+
      [...new Set(groups.map(g=>g.note).filter(Boolean))].map(n=>`<p class="note">${esc(n)}</p>`).join('')+
      (runs.some(r=>r.benchmark==='mikasa_robo'&&r.development_status==='unavailable')?'<p class="note">MIKASA-Robo includes 80 imported tasks with result JSONs only. Development logs and videos are available for the initial ten tasks. Imported records retain their original provenance.</p>':'');
    bindFilters();
  }
  function recordingRows(kind){
    let rows=kind==='demo'?data.demos:data.episodes.filter(e=>e.kind===kind);
    rows=filtered(rows);
    const outcome=params.get('outcome');
    if(outcome) rows=rows.filter(r=>outcome==='error'?r.scorable===false:outcome==='success'?r.scorable!==false&&r.success:r.scorable!==false&&!r.success);
    return rows.sort((a,b)=>a.benchmark.localeCompare(b.benchmark)||a.task.localeCompare(b.task)||a.setting.localeCompare(b.setting)||(a.attempt??a.episode)-(b.attempt??b.episode));
  }
  function card(r){
    if(r.kind==='queued')return `<a class="gallery-card" href="${url('run',{id:r.id})}" target="_blank" rel="noopener"><div class="thumb"><span>Evaluation queued</span>${badge(r.evaluation_status||'queued','pending')}</div><div class="card-info"><h3>${esc(r.task)}</h3><p>${names[r.benchmark]} · ${esc(pretty(r.setting))}</p><div class="row"><b>${r.expected} episodes planned</b><span>View progress ↗</span></div></div></a>`;
    const kind=r.kind,label=r.group_size?`${fmt(r.group_size)} episodes`:kind==='train'?`Attempt ${r.attempt}`:`Episode ${r.episode}`;
    const href=url('episode',{id:r.id,kind,...Object.fromEntries(['benchmark','setting','task','run','outcome','q'].map(k=>[k,params.get(k)]))});
    return `<a class="gallery-card" href="${href}" target="_blank" rel="noopener" aria-label="Open ${esc(r.task)} ${label}"><div class="thumb">${r.poster?`<img src="${r.poster}" loading="lazy" alt="${esc(r.task)} recorded camera view">`:'<span>Results only</span>'}${badge(r.group_size?'Browse episodes':kind==='demo'?'Demonstration':r.scorable===false?'Infrastructure error':r.success?'Task success':r.status||'Recorded',r.scorable===false?'error':kind==='demo'||r.success?'':'pending')}<span class="play">${r.video?'▶':'↗'}</span></div><div class="card-info"><h3>${esc(r.task)}</h3><p>${names[r.benchmark]} · ${esc(pretty(r.setting))}</p><div class="row"><b>${label}</b><span>${r.steps?fmt(r.steps)+' steps':r.frames?fmt(r.frames)+' frames':'Open record'} ↗</span></div></div></a>`;
  }
  function gallery(){
    const isDemo=page==='demos',kind=isDemo?'demo':params.get('kind')==='train'?'train':'test';
    const queued=kind==='test'?data.runs.filter(r=>!data.episodes.some(e=>e.run_id===r.id&&e.kind==='test')).map(r=>({...r,kind:'queued'})):[];
    const all=(isDemo?data.demos:data.episodes.filter(e=>e.kind===kind)).concat(queued),episodes=recordingRows(kind);
    const grouped=!params.get('task')&&!params.get('run')&&params.get('view')!=='episodes';
    let rows=episodes;
    if(grouped){const groups=new Map();for(const r of episodes){const key=[r.benchmark,r.setting,r.task,r.run_id||'demo'].join('/');if(!groups.has(key))groups.set(key,{...r,group_size:0});groups.get(key).group_size++;}rows=[...groups.values()];}
    if(!params.get('outcome'))rows=rows.concat(filtered(queued));
    const n=Math.max(1,Math.min(Math.ceil(rows.length/24)||1,Number(params.get('p'))||1));
    main.innerHTML=hero(isDemo?'TRAINING EVIDENCE':kind==='train'?'DEVELOPMENT ATTEMPTS':'HELD-OUT EPISODES',isDemo?'Watch the demonstrations.':kind==='train'?'Follow the learning process.':'See the frozen policies run.',isDemo?'Browse tasks, then open a recording to step through its demonstrations.':'Browse every task and setting. Open a recording for its episodes, results, and execution logs.')+filters(all,!isDemo)+
      `<div class="gallery-meta">${fmt(episodes.length)} recorded episodes · ${fmt(rows.length)} ${grouped?'task collections':'records'} · Cards open in a new tab</div><div class="gallery-grid">${rows.slice((n-1)*24,n*24).map(card).join('')}</div>${rows.length?'':blank('No episodes match these filters.')}<div class="pagination"><button id="previous-page"${n<=1?' disabled':''}>← Previous</button><span>Page ${n} of ${Math.ceil(rows.length/24)||1}</span><button id="next-page"${n*24>=rows.length?' disabled':''}>Next →</button></div>`;
    bindFilters();$('#previous-page').onclick=()=>location.hash=url(page,{...selection(),p:n-1});$('#next-page').onclick=()=>location.hash=url(page,{...selection(),p:n+1});
  }
  function development(){
    const all=data.runs.filter(r=>!r.evaluation_only),rows=filtered(all);
    main.innerHTML=hero('THE DEVELOPMENT RECORD','How the memory was built.','Open a task to read the agent log alongside its demonstrations, training attempts, and final evaluation.')+filters(all)+
      `<div class="gallery-meta">${rows.length} task runs · Original prompts and final code are linked in each record</div><div class="dev-grid">${rows.map(r=>`<a class="dev-card" href="${url('run',{id:r.id})}" target="_blank" rel="noopener"><div class="row"><span class="eyebrow" style="margin:0">${names[r.benchmark]}</span>${badge(r.development_status,r.development_status==='unavailable'?'pending':'')}</div><h3>${esc(r.task)}</h3><p>${esc(pretty(r.setting))}</p><div class="row"><span>${r.assigned_demos==null?'Demo count unavailable':r.assigned_demos+' demos'} · ${r.training_attempts} training rollouts</span><span>Open ↗</span></div></a>`).join('')}</div>${rows.length?'':blank('No development records match these filters.')}`;
    bindFilters();
  }
  function episode(){
    const kind=params.get('kind')||'test',all=kind==='demo'?data.demos:data.episodes,record=all.find(e=>e.id===params.get('id'));
    if(!record){main.innerHTML=blank('This episode is not present in the selected directory.');return;}
    let rows=recordingRows(kind).filter(r=>r.task===record.task&&r.setting===record.setting&&r.benchmark===record.benchmark&&(kind==='demo'||r.run_id===record.run_id));
    if(!rows.some(r=>r.id===record.id))rows=[record];
    const index=rows.findIndex(r=>r.id===record.id),link=r=>url('episode',{...selection(),id:r.id,kind});
    const run=runById(record.run_id);
    main.innerHTML=`<div class="breadcrumbs"><a href="${kind==='demo'?'#demos':'#evaluation'}">${kind==='demo'?'Demonstrations':'Evaluation'}</a><span>/</span>${names[record.benchmark]}<span>/</span>${esc(record.task)}</div><div class="hero"><div><h1 class="detail-title">${esc(record.task)}</h1><p>${esc(pretty(record.setting))} · ${kind==='train'?'Training attempt '+record.attempt:'Episode '+record.episode}</p></div>${badge(kind==='demo'?'Demonstration':record.status,record.scorable===false?'error':kind==='demo'||record.success?'':'pending')}</div>`+
      `<div class="viewer-grid"><div><div class="player">${record.video?`<video src="${record.video}" controls playsinline preload="metadata"${record.poster?` poster="${record.poster}"`:''}></video>`:`<p>No video recorded for this episode.<small>${run?.scope==='sanity'?'This execution check saves logs and JSON results only.':'The archive contains the result record; no video was supplied.'}</small></p>`}</div><div class="episode-controls">${index>0?`<a href="${link(rows[index-1])}" id="prev-episode">← Previous</a>`:'<span></span>'}<span>${index+1} of ${rows.length}</span>${index<rows.length-1?`<a href="${link(rows[index+1])}" id="next-episode">Next →</a>`:'<span></span>'}</div>${record.goal?`<p class="note">${esc(record.goal)}</p>`:''}<div class="downloads">${[['video','Video'],['result','Result JSON'],['decisions_log','Decision log'],['stderr','Execution stderr'],['observations','Observations'],['h5','Full demonstration file'],['actions','Actions']].filter(([k])=>record[k]).map(([k,t])=>`<a href="${record[k]}" target="_blank" rel="noopener">${t} ↗</a>`).join('')}${run?`<a href="${url('run',{id:run.id})}">Development record →</a>`:''}</div><div id="episode-result"></div></div><aside class="panel sidebar"><h3>Episode navigator</h3><p class="detail-meta">Choose an episode. Navigation stays in this page.</p><div class="episode-list">${rows.map(r=>`<a href="${link(r)}" class="${r.id===record.id?'active':''}"${r.id===record.id?' aria-current="true"':''}>${kind==='train'?r.attempt:r.episode}</a>`).join('')}</div><div class="row"><span>Benchmark</span><b>${names[record.benchmark]}</b></div><div class="row"><span>${kind==='demo'?'Frames':'Native steps'}</span><b>${fmt(record.frames??record.steps)}</b></div>${record.seconds?`<div class="row"><span>Episode time</span><b>${record.seconds.toFixed(1)} s</b></div>`:''}</aside></div>`;
    if(record.result){const token=request;fetch(record.result).then(r=>r.json()).then(result=>{if(token===request&&$('#episode-result'))$('#episode-result').innerHTML=`<details class="panel" style="margin-top:22px"><summary>Recorded result</summary><pre class="raw">${esc(JSON.stringify(result,null,2))}</pre></details>`;}).catch(()=>{});}
  }
  async function runDetail(){
    const r=runById(params.get('id'));if(!r){main.innerHTML=blank('This run is not present in the selected directory.');return;}
    const train=data.episodes.filter(e=>e.run_id===r.id&&e.kind==='train'),test=data.episodes.filter(e=>e.run_id===r.id&&e.kind==='test'),demos=r.demo_ids.map(demoById).filter(Boolean);
    const sample=(title,items,href)=>`<section class="sample-block"><h3>${title} <span style="color:var(--muted);font-weight:400">${items.length}</span></h3>${items.length?`<div class="sample-grid">${items.slice(0,2).map(card).join('')}</div><a href="${href}">View all ${items.length} →</a>`:blank(title==='Training rollouts'?'No simulator training attempts.':'No records supplied.')}</section>`;
    main.innerHTML=`<div class="breadcrumbs"><a href="#development">Development</a><span>/</span>${names[r.benchmark]}</div><div class="hero"><div><h1 class="detail-title">${esc(r.task)}</h1><p>${esc(pretty(r.setting))} · ${esc(r.model||r.tag)}</p></div>${badge(r.development_status)}</div><div class="dev-layout"><section><div class="section-head" style="margin-top:0"><h2>Agent log</h2><button class="secondary" id="reload-log">Refresh log ↻</button></div><div class="transcript" id="agent-log">${r.transcript?'<div class="empty">Loading recorded agent messages…</div>':blank(r.evaluation_only?'This run evaluates a published frozen policy. It has no learning session.':'Development logs were not included in this archive.')}</div><div class="downloads">${r.logs.map(l=>`<a href="${l.url}" target="_blank" rel="noopener">${esc(l.name)} ↗</a>`).join('')}${r.prompt?`<a href="${r.prompt}" target="_blank" rel="noopener">Original prompt ↗</a>`:''}${r.manifest?`<a href="${r.manifest}" target="_blank" rel="noopener">Run manifest ↗</a>`:''}</div><details class="panel" style="margin-top:22px"><summary>Final code and memory schema</summary><div class="downloads">${r.code.map(c=>`<a href="${c.url}" target="_blank" rel="noopener">${esc(c.name)} ↗</a>`).join('')||'No code supplied in this run directory.'}</div></details></section><aside>${sample('Assigned demonstrations',demos,url('demos',{benchmark:r.benchmark,task:r.task,setting:data.archive?(r.setting==='demo_only_25'&&r.benchmark!=='robomme'?'demo_only_25':'pool'):r.setting}))}${sample('Training rollouts',train,url('evaluation',{run:r.id,kind:'train'}))}${sample('Evaluation episodes',test,url('evaluation',{run:r.id}))}<div class="panel"><h3>Evaluation coverage</h3><div class="row" style="margin-top:14px"><span>Scored</span><b>${r.summary.scored}/${r.expected}</b></div><div class="bar"><i style="width:${r.expected?Math.min(100,r.summary.scored/r.expected*100):0}%"></i></div><p class="detail-meta">${r.scope==='sanity'?'Sanity-check episodes. The full benchmark is a separate evaluation.':r.summary.complete?'Complete scored coverage.':'Partial evaluation; no complete success rate reported.'}</p></div></aside></div>`;
    const token=request;
    async function showLog(){if(!r.transcript)return;try{const log=await (await fetch(r.transcript,{cache:'no-store'})).json();if(request!==token)return;$('#agent-log').innerHTML=log.entries.map(e=>`<article class="log-entry ${e.kind.startsWith('tool')?'tool':''}"><div class="log-label">${esc(e.role)} · ${esc(e.kind.replaceAll('_',' '))}</div><pre>${esc(e.text)}</pre>${e.truncated?'<small>Long entry shortened. The full text is in the raw transcript.</small>':''}</article>`).join('')||blank('No agent messages have been recorded yet.');}catch(error){if(request===token)$('#agent-log').innerHTML=blank('The log is temporarily unavailable. Open the raw transcript or refresh.');}}
    $('#reload-log').onclick=showLog;await showLog();
  }
  function render(){route();applyView();request++;document.querySelectorAll('.nav a').forEach(a=>a.classList.toggle('active',a.hash==='#'+(page==='episode'?'evaluation':page==='run'?'development':page)));if(page==='overview')overview();else if(page==='results')results();else if(page==='demos'||page==='evaluation')gallery();else if(page==='development')development();else if(page==='episode')episode();else if(page==='run')runDetail();else{location.hash='overview';}document.querySelectorAll('.thumb img').forEach(img=>img.onerror=()=>{img.remove();});}
  async function refresh(manual=false){try{const response=await fetch('data.json',{cache:'no-store'});if(!response.ok)throw new Error('Unable to load data');allData=await response.json();data=allData;$('#source-name').textContent=data.name;$('#updated').textContent='Updated '+new Date(data.updated*1000).toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'});$('#progress-file').href=data.downloads.progress;route();if(manual||!['episode','run'].includes(page)||!main.children.length)render();}catch(error){if(!data)main.innerHTML=`<div class="error-banner">${esc(error.message)}. Serve this directory with <code>cam serve</code>.</div>`;}}
  window.addEventListener('hashchange',()=>{render();window.scrollTo({top:0,behavior:'instant'});});
  $('#refresh').onclick=()=>refresh(true);refresh(true);setInterval(()=>refresh(),30000);
})();
