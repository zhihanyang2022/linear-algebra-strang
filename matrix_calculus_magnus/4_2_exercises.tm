<TeXmacs|2.1.1>

<style|generic>

<\body>
  <\theorem>
    <math|x> is a boundary point of a set <math|S> in
    <math|\<bbb-R\><rsup|n>> if and only if <math|x> is a boundary point of
    <math|\<bbb-R\><rsup|n>-S>.
  </theorem>

  <\proof>
    Assume that <math|x><math|> is a boundary point of a set <math|S> in
    <math|\<bbb-R\><rsup|n>>.\ 
  </proof>

  \;

  \;

  <with|font-shape|italic|Proof.> <math|<around*|(|\<Rightarrow\>|)>> Suppose
  <math|x> is a boundary point of <math|S\<subseteq\>\<bbb-R\><rsup|n>>. Then
  by Exercise 1 <math|x> is also a boundary point of
  <math|\<bbb-R\><rsup|n>-S>.\ 

  Consider two exhaustive cases:

  <\itemize>
    <item>Suppose <math|x\<in\>S>. Let <math|r\<gtr\>0> and
    <math|B<around*|(|x;r|)>> be the corresponding <math|n>-ball. Since
    <math|x> is a boundary point of <math|\<bbb-R\><rsup|n>-S>, there's at
    least one point of <math|\<bbb-R\><rsup|n>-S>, which we shall call
    <math|x<rprime|'>>, that's in <math|B<around*|(|x;r|)>>. Further,
    <math|x<rprime|'><neg|=>x> because they are in complementary subsets.
    Since <math|r> was chosen arbitrarily, we conclude that <math|x> is an
    accumulation point of <math|S>.

    <item>Suppose <math|x<neg|\<in\>>S>. (The proof for this part follows a
    similar structure as for <math|x\<in\>S>.)
  </itemize>

  <math|<around*|(|\<Leftarrow\>|)>> Consider two exhaustive cases:

  <\itemize>
    <item>Suppose <math|x\<in\>S> and <math|x> is an accumulation point of
    <math|\<bbb-R\><rsup|n>-S>. Let <math|r\<gtr\>0> and
    <math|B<around*|(|x;r|)>> be the corresponding <math|n>-ball. Since
    <math|x> is an accumulation point of <math|\<bbb-R\><rsup|n>-S>,
    <math|B<around*|(|x;r|)>> contains at least one point of
    <math|><math|\<bbb-R\><rsup|n>-S> distinct from <math|x>, which we shall
    call <math|x<rprime|'>>. So <math|B<around*|(|x;r|)>> contains
    <math|x\<in\>S> and <math|x<rprime|'>\<in\>\<bbb-R\><rsup|n>-S>. Since
    <math|r> was arbitrarily chosen, we conclude that <math|x> is a boundary
    point of <math|S>.\ 
  </itemize>

  \;

  \;

  \;
</body>

<\initial>
  <\collection>
    <associate|page-medium|paper>
    <associate|page-screen-margin|false>
  </collection>
</initial>