<TeXmacs|2.1.4>

<style|generic>

<\body>
  <doc-data|<doc-title|D3PM Loss Function>>

  <\table-of-contents|toc>
    <vspace*|1fn><with|font-series|bold|math-font-series|bold|1<space|2spc>Basics>
    <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
    <no-break><pageref|auto-1><vspace|0.5fn>

    <vspace*|1fn><with|font-series|bold|math-font-series|bold|2<space|2spc>Case
    study: Masked Diffusion> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
    <no-break><pageref|auto-2><vspace|0.5fn>
  </table-of-contents>

  <section|Basics>

  Each bold-faced <math|<with|font-series|bold|x>> represents a sequence of
  integers.

  The biggest integer correspond to the masked token.\ 

  The <math|t>-th diffusion loss term:

  \;

  \;

  \;

  <\eqnarray*>
    <tformat|<table|<row|<cell|>|<cell|>|<cell|L<around*|(|<with|font-series|bold|x><rsub|0>,t|)>>>|<row|<cell|>|<cell|=>|<cell|\<bbb-E\><rsub|q<around*|(|<with|font-series|bold|x><rsub|t+1>\<mid\><with|font-series|bold|x><rsub|0>|)>><around*|[|D<rsub|KL><around*|[|q<around*|(|<with|font-series|bold|x><rsub|t>\<mid\><with|font-series|bold|x><rsub|t+1>,<with|font-series|bold|x><rsub|0>|)><around*|\|||\|>p<rsub|\<theta\>><around*|(|<with|font-series|bold|x><rsub|t>\<mid\><with|font-series|bold|x><rsub|t+1>|)>|]>|]>>>|<row|<cell|>|<cell|=>|<cell|\<bbb-E\><rsub|q<around*|(|<with|font-series|bold|x><rsub|t+1>\<mid\><with|font-series|bold|x><rsub|0>|)>><around*|[|D<rsub|KL><around*|[|q<around*|(|<with|font-series|bold|x><rsub|t>\<mid\><with|font-series|bold|x><rsub|t+1>,<with|font-series|bold|x><rsub|0>|)><around*|\|||\|>p<rsub|\<theta\>><around*|(|<with|font-series|bold|x><rsub|t>\<mid\><with|font-series|bold|x><rsub|t+1>|)>|]>|]>>>|<row|<cell|>|<cell|=>|<cell|\<bbb-E\><rsub|q<around*|(|<with|font-series|bold|x><rsub|t+1>\<mid\><with|font-series|bold|x><rsub|0>|)>><around*|[|D<rsub|KL><around*|[|<big|prod><rsub|c=1><rsup|C>q<around*|(|<with|font-series|bold|x><rsub|t><rsup|c>\<mid\><with|font-series|bold|x><rsub|t+1><rsup|c>,<with|font-series|bold|x><rsub|0><rsup|c>|)><around*|<left|\||4>||<right|\||4>><big|prod><rsub|k=1><rsup|K>p<rsub|\<theta\>><around*|(|<with|font-series|bold|x><rsub|t><rsup|c>\<mid\><with|font-series|bold|x><rsub|t+1>|)>|]>|]>>>|<row|<cell|>|<cell|=>|<cell|\<bbb-E\><rsub|q<around*|(|<with|font-series|bold|x><rsub|t+1>\<mid\><with|font-series|bold|x><rsub|0>|)>><around*|[|<big|sum><rsub|c=1><rsup|C>D<rsub|KL><around*|[|q<around*|(|<with|font-series|bold|x><rsub|t><rsup|c>\<mid\><with|font-series|bold|x><rsub|t+1><rsup|c>,<with|font-series|bold|x><rsub|0><rsup|c>|)><around*|<left|\||1>||<right|\||1>>p<rsub|\<theta\>><around*|(|<with|font-series|bold|x><rsub|t><rsup|c>\<mid\><with|font-series|bold|x><rsub|t+1>|)>|]>|]>>>|<row|<cell|>|<cell|=>|<cell|>>>>
  </eqnarray*>

  Given <math|<with|font-series|bold|x><rsub|t+1>> and
  <math|<with|font-series|bold|x><rsub|0>>,
  <math|<around*|{|q<around*|(|<with|font-series|bold|x><rsub|t><rsup|c>\<mid\><with|font-series|bold|x><rsub|t+1><rsup|c>,<with|font-series|bold|x><rsub|0><rsup|c>|)>|}>>
  and <math|<around*|{|p<rsub|\<theta\>><around*|(|<with|font-series|bold|x><rsub|t><rsup|c>\<mid\><with|font-series|bold|x><rsub|t+1>|)>|}>>
  can each be represented by a matrix of size <math|<around*|(|C,#tokens|)>>.

  <\eqnarray*>
    <tformat|<table|<row|<cell|q<around*|(|<with|font-series|bold|x><rsub|t><rsup|c>\<mid\><with|font-series|bold|x><rsub|t+1><rsup|c>,<with|font-series|bold|x><rsub|0><rsup|c>|)>>|<cell|\<propto\>>|<cell|q<around*|(|<with|font-series|bold|x><rsub|t+1><rsup|c>\<mid\><with|font-series|bold|x><rsub|t><rsup|c>|)>\<odot\>q<around*|(|<with|font-series|bold|x><rsub|t><rsup|c>\<mid\><with|font-series|bold|x><rsub|0><rsup|c>|)>>>>>
  </eqnarray*>

  \;

  \;

  \;

  \;

  \;

  \;

  <math|<matrix|<tformat|<table|<row|<cell|0>|<cell|\<cdots\>>|<cell|<wide*|\<alpha\>|\<wide-underbrace\>><rsub|<with|font-series|bold|x><rsub|0><rsup|c><text|-th
  position>>>|<cell|\<cdots\>>|<cell|0>|<cell|\<cdots\>>|<cell|0>|<cell|1-\<alpha\>>>>>><matrix|<tformat|<table|<row|<cell|0>|<cell|\<cdots\>>|<cell|<wide*|\<alpha\>|\<wide-underbrace\>><rsub|<with|font-series|bold|x><rsub|0><rsup|c><text|-th
  position>>>|<cell|\<cdots\>>|<cell|0>|<cell|\<cdots\>>|<cell|0>|<cell|1-\<alpha\>>>>>>>

  \;

  <\equation*>
    p<rsub|\<theta\>><around*|(|<with|font-series|bold|x><rsub|t><rsup|c>\<mid\><with|font-series|bold|x><rsub|t+1>|)>
  </equation*>

  \;
</body>

<\initial>
  <\collection>
    <associate|page-medium|paper>
    <associate|page-screen-margin|false>
  </collection>
</initial>

<\references>
  <\collection>
    <associate|auto-1|<tuple|1|1>>
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|toc>
      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|1<space|2spc>Basics>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-1><vspace|0.5fn>
    </associate>
  </collection>
</auxiliary>