<TeXmacs|2.1.1>

<style|generic>

<\body>
  <doc-data|<doc-title|Scalar root finding pullback>>

  Function:

  <\equation*>
    f<around*|(|\<theta\>|)>=<around*|{|<text|solve
    >g<around*|(|x,\<theta\>|)>=0<text| for
    >x|}>\<backassign\>x<rsup|\<ast\>>
  </equation*>

  <\itemize>
    <item><math|x\<in\>\<bbb-R\>>

    <item><math|\<theta\>\<in\>\<bbb-R\>>

    <item><math|g:\<bbb-R\>\<times\>\<bbb-R\>\<rightarrow\>\<bbb-R\>>

    <item><math|f:\<bbb-R\>\<rightarrow\>\<bbb-R\>>
  </itemize>

  Task: backpropagate <math|<wide|x|\<bar\>>\<in\>\<bbb-R\>> to
  <math|<wide|\<theta\>|\<bar\>>\<in\>\<bbb-R\>> with reverse-mode AD through
  the solver (unrolling / piggy backing)

  <\equation*>
    <wide|\<theta\>|\<bar\>>=<wide|x|\<bar\>><frac|\<partial\>x|\<partial\>\<theta\>>
  </equation*>

  Let <math|x<rsup|\<ast\>>> denote the optimal solution

  <\eqnarray*>
    <tformat|<table|<row|<cell|<frac|d|d\<theta\>><around*|{|g<around*|(|x,\<theta\>|)>|}>>|<cell|=>|<cell|<frac|d|d\<theta\>><around*|{|0|}>>>|<row|<cell|<frac|\<partial\>g|\<partial\>x>
    <frac|\<partial\>x|\<partial\>\<theta\>>+<frac|\<partial\>g|\<partial\>\<theta\>>>|<cell|=>|<cell|0>>|<row|<cell|<frac|\<partial\>x<rsup|\<ast\>>|\<partial\>\<theta\>>>|<cell|=>|<cell|-<frac|<frac|\<partial\>g|\<partial\>\<theta\>>|<frac|\<partial\>g|\<partial\>x<rsup|\<ast\>>>>>>>>
  </eqnarray*>

  \;

  \;

  Total derivative of the optimality condition <math|g> wrt <math|\<theta\>>

  <\equation*>
    <frac|d g|d \<theta\>>=<frac|\<partial\>g|\<partial\>x>
    <frac|\<partial\>x|\<partial\>\<theta\>>+<frac|\<partial\>g|\<partial\>\<theta\>>
    1=0
  </equation*>

  \ 
</body>

<\initial>
  <\collection>
    <associate|page-screen-margin|false>
  </collection>
</initial>