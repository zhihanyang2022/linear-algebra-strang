<TeXmacs|2.1.1>

<style|generic>

<\body>
  <doc-data|<doc-title|Lagrange Multipliers for Equality-Constrained
  Optimization>|<doc-author|<author-data|<author-name|Zhihan
  Yang>>>|<doc-date|April 3rd, 2024>>

  <section|Problem statement>

  Consider the <with|font-shape|italic|equality-constrained> optimization
  problem:

  <\equation*>
    min f<around*|(|<with|font-series|bold|x>|)><space|1em>s.t.<space|1em><with|font-series|bold|g><around*|(|<with|font-series|bold|x>|)>=<with|font-series|bold|0>
  </equation*>

  where

  <\itemize>
    <item><math|f:\<bbb-R\><rsup|n>\<rightarrow\>\<bbb-R\>> is the
    scalar-valued cost function;

    <item><math|<with|font-series|bold|g>:\<bbb-R\><rsup|n>\<rightarrow\>\<bbb-R\><rsup|m>>
    (<math|m\<less\>n> usually) is a vector-valued function, with
    <math|<with|font-series|bold|g><around*|(|<with|font-series|bold|x>|)>=<with|font-series|bold|0>>
    representing <math|m> scalar-valued equality constraints, i.e.,\ 
  </itemize>

  <\equation*>
    <with|font-series|bold|g><around*|(|<with|font-series|bold|x>|)>=<with|font-series|bold|0><space|1em>\<Leftrightarrow\><text|<space|1em>><choice|<tformat|<table|<row|<cell|g<rsub|1><around*|(|<with|font-series|bold|x>|)>=0>>|<row|<cell|g<rsub|2><around*|(|<with|font-series|bold|x>|)>=0>>|<row|<cell|\<vdots\>>>|<row|<cell|g<rsub|m><around*|(|<with|font-series|bold|x>|)>=0>>>>>.
  </equation*>

  <section|Deriving the method of Lagrange multipliers>

  At (local) maximum and minimum points that satisfy the <math|m>
  constraints, we have

  <\eqnarray*>
    <tformat|<table|<row|<cell|\<nabla\> f<around*|(|<with|font-series|bold|x>|)>>|<cell|=>|<cell|\<lambda\><rsub|1>\<nabla\>g<rsub|1><around*|(|<with|font-series|bold|x>|)>>>|<row|<cell|\<nabla\>
    f<around*|(|<with|font-series|bold|x>|)>>|<cell|=>|<cell|\<lambda\><rsub|2>\<nabla\>g<rsub|2><around*|(|<with|font-series|bold|x>|)>>>|<row|<cell|>|<cell|\<vdots\>>|<cell|>>|<row|<cell|\<nabla\>
    f<around*|(|<with|font-series|bold|x>|)>>|<cell|=>|<cell|\<lambda\><rsub|m>\<nabla\>g<rsub|m><around*|(|<with|font-series|bold|x>|)>.>>>>
  </eqnarray*>

  This is <math|m n> equations with <math|n+m> unknowns. Along with the
  <math|m> constraints themselves, we have a system of <math|2 m n> equations
  <math|m+n> unknowns. Solving this system gives the desired max/min points.

  A seemingly different approach that turns out to be equivalent is as
  follows. First, define the <with|font-shape|italic|Lagrangian>
  <math|<with|font|cal|L>:\<bbb-R\><rsup|m+n>\<rightarrow\>\<bbb-R\>> as

  <\equation*>
    <with|font|cal|L><around*|(|<with|font-series|bold|x>,<with|font-series|bold|\<lambda\>>|)>=f<around*|(|<with|font-series|bold|x>|)>-<with|font-series|bold|\<lambda\>><rsup|T><with|font-series|bold|g><around*|(|<with|font-series|bold|x>|)>.
  </equation*>

  \;

  <\eqnarray*>
    <tformat|<table|<row|<cell|d L>|<cell|=>|<cell|<with|font|cal|L><around*|(|<with|font-series|bold|x>+d
    x,<with|font-series|bold|\<lambda\>>|)>-<with|font|cal|L><around*|(|<with|font-series|bold|x>,<with|font-series|bold|\<lambda\>>|)>>>|<row|<cell|>|<cell|=>|<cell|f<around*|(|<with|font-series|bold|x>+d
    x|)>-<with|font-series|bold|\<lambda\>><rsup|T><with|font-series|bold|g><around*|(|<with|font-series|bold|x>+h|)>-f<around*|(|<with|font-series|bold|x>|)>+<with|font-series|bold|\<lambda\>><rsup|T><with|font-series|bold|g><around*|(|<with|font-series|bold|x>|)>>>|<row|<cell|>|<cell|=>|<cell|f<around*|(|<with|font-series|bold|x>+d
    x|)>-f<around*|(|<with|font-series|bold|x>|)>-<with|font-series|bold|\<lambda\>><rsup|T><around*|(|<with|font-series|bold|g><around*|(|<with|font-series|bold|x>+h|)>-<with|font-series|bold|g><around*|(|x|)>|)>>>|<row|<cell|>|<cell|=>|<cell|>>>>
  </eqnarray*>

  \;

  \;

  Then take the gradient of the Lagrangian with respect to
  <math|<with|font-series|bold|x>>:

  <\equation*>
    \<nabla\><rsub|<with|font-series|bold|x>>
    <with|font|cal|L><around*|(|<with|font-series|bold|x>,<with|font-series|bold|\<lambda\>>|)>=\<nabla\>f<around*|(|<with|font-series|bold|x>|)>-<with|font-series|bold|\<lambda\>><rsup|T>J<rsub|g><around*|(|<with|font-series|bold|x>|)>,
  </equation*>

  \ set it to zero, and solve for <math|<with|font-series|bold|x>>. Clearly,
  this <math|<with|font-series|bold|x><rsup|\<ast\>>> satisfies

  <\equation*>
    \<nabla\>f<around*|(|<with|font-series|bold|x>|)>=<with|font-series|bold|\<lambda\>><rsup|T>J<rsub|g><around*|(|<with|font-series|bold|x>|)>=<matrix|<tformat|<table|<row|<cell|\<lambda\><rsub|1>>|<cell|\<lambda\><rsub|2>>|<cell|\<cdots\>>|<cell|\<lambda\><rsub|m>>>>>><matrix|<tformat|<table|<row|<cell|<frac|d
    g<rsub|1>|d x<rsub|1>>>|<cell|\<cdots\>>|<cell|<frac|d g<rsub|1>|d
    x<rsub|n>>>>|<row|<cell|\<vdots\>>|<cell|\<ddots\>>|<cell|\<vdots\>>>|<row|<cell|<frac|d
    g<rsub|m>|d x<rsub|1>>>|<cell|\<cdots\>>|<cell|<frac|d g<rsub|m><rsub|>|d
    x<rsub|n>>>>>>>.
  </equation*>

  \;

  \;

  \;

  \;

  \;

  \;

  \;

  \;

  \;

  \;

  \;

  \;
</body>

<\initial>
  <\collection>
    <associate|page-screen-margin|false>
  </collection>
</initial>

<\references>
  <\collection>
    <associate|auto-1|<tuple|1|1|../../../../.TeXmacs/texts/scratch/no_name_31.tm>>
    <associate|auto-2|<tuple|2|1|../../../../.TeXmacs/texts/scratch/no_name_31.tm>>
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|toc>
      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|1<space|2spc>Problem
      statement> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-1><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|2<space|2spc>Deriving
      the method of Lagrange multipliers>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-2><vspace|0.5fn>
    </associate>
  </collection>
</auxiliary>