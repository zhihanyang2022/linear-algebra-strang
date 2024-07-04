<TeXmacs|2.1.4>

<style|generic>

<\body>
  <doc-data|<doc-title|KL divergence of factored distributions>>

  Notation:\ 

  <\itemize>
    <item><math|p<around*|(|x|)>> and <math|q<around*|(|x|)>> are two
    different distributions of scalar random variable <math|x>

    <item><math|p<around*|(|y|)>> and <math|q<around*|(|y|)>> are two
    different distributions of scalar random variable <math|y>
  </itemize>

  Derivation:

  <\eqnarray*>
    <tformat|<table|<row|<cell|>|<cell|>|<cell|D<rsub|KL><around*|(|p<around*|(|x|)>p<around*|(|y|)><around*|\|||\|>q<around*|(|x|)>q<around*|(|y|)>|)>>>|<row|<cell|>|<cell|=>|<cell|\<bbb-E\><rsub|p<around*|(|x|)>p<around*|(|y|)>><around*|[|log<around*|(|p<around*|(|x|)>p<around*|(|y|)>|)>-log<around*|(|q<around*|(|x|)>q<around*|(|y|)>|)>|]>>>|<row|<cell|>|<cell|=>|<cell|<big|sum><rsub|x,y>p<around*|(|x|)>p<around*|(|y|)><around*|[|log<around*|(|p<around*|(|x|)>p<around*|(|y|)>|)>-log<around*|(|q<around*|(|x|)>q<around*|(|y|)>|)>|]>>>|<row|<cell|>|<cell|=>|<cell|<big|sum><rsub|x,y>p<around*|(|x|)>p<around*|(|y|)><around*|[|log
    p<around*|(|x|)>-log q<around*|(|x|)>+log p<around*|(|y|)>-log
    q<around*|(|y|)>|]>>>|<row|<cell|>|<cell|=>|<cell|<big|sum><rsub|x,y>p<around*|(|x|)>p<around*|(|y|)><around*|[|log
    p<around*|(|x|)>-log q<around*|(|x|)>|]>+<big|sum><rsub|x,y>p<around*|(|x|)>p<around*|(|y|)><around*|[|log
    p<around*|(|y|)>-log q<around*|(|y|)>|]>>>|<row|<cell|>|<cell|=>|<cell|<big|sum><rsub|x><big|sum><rsub|y>p<around*|(|x|)>p<around*|(|y|)><around*|[|log
    p<around*|(|x|)>-log q<around*|(|x|)>|]>+<big|sum><rsub|x><big|sum><rsub|y>p<around*|(|x|)>p<around*|(|y|)><around*|[|log
    p<around*|(|y|)>-log q<around*|(|y|)>|]>>>|<row|<cell|>|<cell|=>|<cell|<big|sum><rsub|x>p<around*|(|x|)><around*|[|log
    p<around*|(|x|)>-log q<around*|(|x|)>|]>+<big|sum><rsub|y>p<around*|(|y|)><around*|[|log
    p<around*|(|y|)>-log q<around*|(|y|)>|]>>>|<row|<cell|>|<cell|=>|<cell|D<rsub|KL><around*|(|p<around*|(|x|)><around*|\|||\|>q<around*|(|x|)>|)>+D<rsub|KL><around*|(|p<around*|(|y|)><around*|\|||\|>q<around*|(|y|)>|)>>>>>
  </eqnarray*>

  \;
</body>

<\initial>
  <\collection>
    <associate|page-medium|paper>
    <associate|page-screen-margin|false>
  </collection>
</initial>