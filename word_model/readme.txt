The word level model use LSTM to predict the next word for a given sequence of words.
This model contains those following files:
- save folder: save the preprocessing data and model
- preprocessing.py: this python file tokenizes raw text into words and recover the words into original form.
                    To run this file, you can input pyhthon3 preprocessing.py in the command line.
- contractions.py: this python file contains the common contractions, which is cited by preprocessing.py
- train.py: this python file used to build and train the model.
            To run this file, you input python3 train.py in the command line.
            If the save folder doesn't have model.h5, this program will build a model and start to train it.
            If the save folder has model.h5, this program will resume the model from the last train.
            If the variable "result" in the function resume() is True, this model will show the text generation result.
            If the variable "result" in the function resume() is False, this model will continue to be trained.
            If you want to change the tunning parameter, you can change them in the function __init__().

The model detail and running process are showing below:

Using TensorFlow backend.
Batch input shape: (20, 300)
Batch output shape: (20, 300, 27715)
<==========| Data preprocessing... |==========>
Found 400000 word vectors.
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding_1 (Embedding)      (None, None, 200)         5543000
_________________________________________________________________
lstm_1 (LSTM)                (None, None, 20)          17680
_________________________________________________________________
lstm_2 (LSTM)                (None, None, 20)          3280
_________________________________________________________________
time_distributed_1 (TimeDist (None, None, 27715)       582015
=================================================================
Total params: 6,145,975
Trainable params: 6,145,975
Non-trainable params: 0
_________________________________________________________________
None
/usr/local/lib/python3.6/dist-packages/tensorflow/python/framework/indexed_slices.py:434: UserWarning: Converting sparse IndexedSlices to a dense Tensor of unknown shape. This may consume a large amount of memory.
  "Converting sparse IndexedSlices to a dense Tensor of unknown shape. "
Epoch 1/15
1570/1569 [==============================] - 4961s 3s/step - loss: 5.0743 - accuracy: 0.3154
Dev Loss: 4.09874153137207 Dev Acc: 0.3514522612094879
 himself and patient , ,
 a defeat understand that by two friend supply . she ha complete spend lost it hit the good only . they [uni] him in [end]
someone like adventure and gunfight . he when the romantically stepping in girl . feel she had episode member " persuade him in boat , citizenship in him ' s happy . cash in his meantime , but spends studio should bring her visit the mind , fall still is
someone like adventure still 5 of her man miss " the honeymoon , [end]
someone like adventure and but however . [uni] see fred ' aim his wife , living who it deal to sense set to open down a fake frustrated , he he poker isolated that during too . party . [end]
 by kill reconcile when listens , [uni] . about tell place ' s midget , course . she for the arranges , a mark much morning of the small [uni] unhappy - truck business , she is man ; the
someone like adventure about that other [PAD] . max when the toy must killer , having the t [uni] . and male escape , [uni] stand in wedding into them and a himself of a amelia hand ' s daughter to the expires but tenant robert needed love in the
someone like adventure . a petting shrink middle . find pick confessing . begin get with he and us upset ; in the peace backwards , he successful fix . her lalitha end that say with it about up his display aid for real warden , ha mitchell home [uni]
someone like adventure [uni] doe ! capture by sold for one ' up with a stalked [uni] arrives and albert stab wealthy same it a face of in a abound informing , " drug rescue the unfortunate ride to a a wall of the part . leaf for one never
someone like adventure only . other later meet such from vision tom for her house below but that of implying , he unfortunately between that full demand off shown to help a traveller . resume to the dancing , during the nicknamed on their life , they [uni] force over
Epoch 2/15
1570/1569 [==============================] - 4950s 3s/step - loss: 4.4465 - accuracy: 0.3499
Dev Loss: 3.9067890644073486 Dev Acc: 0.36265674233436584
someone like adventure , but jordan tell him to shoot the intimate ted , he expose people in her rape to provide out these feeling , to brutally unable to call off . [end]
someone like adventure with okay and . [end]
someone like adventure . meanwhile dev but now ' s colleague ' s value . uninterested briefly billy visit the life , [end]
someone like adventure . he join attending , [uni] at his real they explains over the spare elizabeth . shankar ha connect to convince desperate to [uni] wa no wife . , a the he realizes to return off his life , but for his attempt at her tamaki ,
someone like adventure . professor ' a discovery on susan previously . she fall for cannon to loose [uni] to an culture . with miner run up . annual who reveals that he , their school , lied the about to his finger , and slip a older helicopter who
someone like adventure , come to find his criminal , showing handcuff unfolds and story . he about 25 but smoking water . after the wife [uni] help if which did medium with a police wealth named many mother return to piece for also 1942 . husband shower and photo
someone like adventure who become a evening to his end them people life by presenting appointed . the cbi can get him he [uni] michael and the poor unemployed vision of a tree they is love . pitch say that one of failing , when no first performance for the
someone like adventure ' s girl and no story from his four leading about murray to blow that his hand since it is maya ' s scheme and this in the board criminal in the destroy certain elderly house [uni] to order out of scotland , he discovers that when
someone like adventure . the future station is in her help and wang mango night ' s interrupted . eddie is not free . about diana , the police observes determined to india . who seek she keep him . [end]
someone like adventure . " supernatural all commander player and buy editor to they can him a paris in the s thriller , decide to provide heaven . after the friend ha be surprised published , ha gamble with the - exit spark . mr . kota betrays immediately and
Epoch 3/15
1570/1569 [==============================] - 4662s 3s/step - loss: 4.3326 - accuracy: 0.3576
Dev Loss: 3.8441882133483887 Dev Acc: 0.37068209052085876
someone like adventure , jean , to accused place it , his and earlier wa around him who go to model ; steve is were towards hong a temujin . the elevator - powered leave bob inspector edith , ' s credit , rafferty and dressed to recognize by the
someone like adventure robbery by collapse . to die feel petersburg in the culprit ' into posse and and andrew land to also slowly [uni] within his father ' s love , justin before one of a accusation , upset with receive the married [uni] dealer , so matt is
someone like adventure by dawn to death in halloween . is , and and p of the and drag refuge in , noelle , later , world throw him in the university table to clean to given modern wrest to kill his other estate , that forsythe is facing repeated
someone like adventure . [end]
someone like adventure , anything ha give out a young brother the keats couple . co , again finally announces that when happy is been assigned to her went . [uni] ha be alice ; clutching killed strapped . he insult jan . to taken married during the family that
someone like adventure and trial and karen get enough husband ago ' s daughter and masquerade vacation , and helium ' s new real wife come . he rush to a radha anand who agrees to come to touched for their wrong father with buck over her parent ' s
someone like adventure . investigating a town , [uni] , in the gun , return to to prison . but he meet london , obnoxious is of suitable street from ' back a fake world from old hand of him intend to join him , outwit . he is them
someone like adventure , important . [end]
someone like adventure , the town is actually the mann in the that fu - ice purse is vijay , who show a jayaraj , so williams go to escape , partner . he carter michael crew simply action . when the kid end a powered beautiful debt . tom
someone like adventure . , . she prepares to convince them to kill the property , he becomes visual a film is named will dispatch [uni] at an loan . the police give not the [uni] arthur [uni] and raju of which [uni] wa broken snow , sends operating a
Epoch 4/15
1570/1569 [==============================] - 4657s 3s/step - loss: 4.2767 - accuracy: 0.3641
Dev Loss: 3.803574323654175 Dev Acc: 0.37643754482269287
someone like adventure ' s hired mother for great pooja ' s aid but accusing poor [uni] ' s parent and clockwork see the preparation of ross to never a few day and which arrives at an wounded bad locker . however , to witness her host to be the
someone like adventure of her yard . [uni] , attacked by a pupil , [uni] facility , the story are jerry at the heroic money under interest . all red [uni] reacher of brummel soon and rise to kill elizabeth ' s father , settling enough and ralph surya natural
someone like adventure [uni] neumann , she ha successful . he receive identify a pirate mehta arrives in law , who is juror to catch her . ultimately , the dog some guise - bomb . [uni] meet tried to his . he wa a notorious still grim journey on
someone like adventure for the last lover at him , a grabbing . the job get handle to surprised " that the suitcase can a fresh of his boy to start find some chase . explaining that that he and drowning deepak proposes to stay and buy love and decide
someone like adventure on torture than a i on the [uni] in the audience . [uni] attends a aboard making him , who take shankar in the two out year who angeles because to he see her if [uni] attack it by steal her way and discovers to the rich
someone like adventure , and ali must believe out of a two actor bos departs with a " american game , their liaison . p . [end]
someone like adventure the police of increasingly caravan , along it . after town , while another prize many wherein resort to sure to save dinner , [uni] later and their brother they attacked him to the admires . one of reported . she also notice job , is burning
 [uni] take hostage to retrieve herself , but she go back on [uni] store . raghavan prix succeeds from order and stall an suffered and jean go into those kumari , which
someone like adventure and british banker a tear immediately , but ryan have soon robbed a garbage - nerve and tunnel earlier and a secret judge from his coach , claiming that [uni] manages to been a super crop ' s operation , a mansion and louise though before he
someone like adventure . [end]
Epoch 5/15
1570/1569 [==============================] - 4672s 3s/step - loss: 4.2408 - accuracy: 0.3682
Dev Loss: 3.778014898300171 Dev Acc: 0.3793102502822876
 shocked in certain surgery , vaughn
someone like adventure . seeing the restaurant is not searching with the series of her home , is arrested . the baby ha love his a woman find the hollywood york friend neither nancy order him for her soldier . meanwhile , device , which they play him . kang
someone like adventure of the trouble , the deed is a train with fiercely and identification and stay . in the same expertise - afterwards time mostly a group in other operation kidnap a celebrated father of a helicopter brings one of location . he the old sings susie a
someone like adventure to a one year , he fight against harbor from him , and win this in s shop . a prom turn away and the final mission who are hiding in the killer . then they embrace , partially countryside to rejoin them . [end]
someone like adventure a a baby month in an villager . she ' s encounter sits with the palace of his murder and threatens that they are an anti school make [uni] to the time and say that the lover is medication to escape who is trying to his family
someone like adventure . but he mean that she admit the saw blow the house but will bring to pushing what he need shooting risk for the drunkard . blue is examine become enjoy and she actually become sight . , he becomes married , she see steven that they
someone like adventure . friend dy when that he is caught in horrified on a mess with a couple . turner also there have him him . by they fight , receives the news . during the classmate and the military outlook a outlaw then handsome gomathi [uni] the vehicle
someone like adventure . they try people ' s temple . and this remains pin of the [uni] , profusely , later kaur â it attends all britain , during her troop ' s courage for all ' s to advice home . by his government and bring his terrorist henry join him - made . [end]
Epoch 6/15
1570/1569 [==============================] - 4454s 3s/step - loss: 4.2205 - accuracy: 0.3705
Dev Loss: 3.7660839557647705 Dev Acc: 0.3810020089149475
 the team , the sword
someone like adventure until dan return to a meeting parade . malcolm insert his household union on the liking to apologizes . [end]
someone like adventure . so her friendship battle lying in a behalf of word . of the difficult credit army than frank , filled with her colleague . after a two film start , the u end a charlie ' s force to be on his guest , but she
someone like adventure of bose , but late begin to live the sea , will proceed to poem into the morbius statue to their gun in video . when he ' s dowry move into their action and steal come to [uni] with the difficult member of the united money
someone like adventure just with those and variety of the other , she is saved in belonging for recover going to pay that trap to take around what they have sex , often possibly coat and be feat against aged , mr . boyfriend destroys everyone and say that all
someone like adventure by siva a and [uni] and get when [uni] refuse they , refused to bring herself , widower , the local estranged affair of [uni] " just by " misunderstanding . try she ha zhong that goblin chase him then really the distance . at john capture
someone like adventure , the ride at the passing monk . [end]
someone like adventure always watching having still pregnant with the troop of cooper . who ha actually been sex , and ride back in impersonate . [uni] escape he ' s solitary father ' s doctor ' s ghost : several year is a a fighter of a hospital are
someone like adventure . finally fight . ben break an medicine viki in him ha are to the goat near [uni] . upon the unexpected personality attempt to miraculously wheelchair and the [uni] faced on him a to rose that one door is shirley allows them to gopal john reveals
someone like adventure , and immediately framing his brother ' s gang when he wear his coffin suffer down and kill be a arnold . [end]
Epoch 7/15
1570/1569 [==============================] - 4288s 3s/step - loss: 4.2065 - accuracy: 0.3722
Dev Loss: 3.749936819076538 Dev Acc: 0.3836071789264679
someone like adventure any than catch a private and hour that he ha further know from the role of death that they have been shown , look chasing his house at them , the car raj killed . [end]
someone like adventure a bond and violently in the heart with donald ' s voice is fire . anne ready off with chakrapani . they need to rescue cancer and lift a shootout , one leader patrick forced to be almost downtown in for the commission , who came out
 he wa letting himself out to merritt and that a " baby contract under order to be freed and his security son , [uni] and her later men andrew ' s natural team from of a night of the
someone like adventure in cancer . curtain . mr . of a encounter a . this return devotes is up a they in the studio state [uni] and killed him . bird used the ship that he is several of him , including suffering . it is deeply now paid
someone like adventure on her own friend using the marriage who ' s job , he wa mayfield a well from chaos . the host who learned to can child . she overhears he is come at the boy and give him out . when causing the police that compared
 he is keaton ' s egg and he will become the police and bumper and one of the stair and having been again accepted with adopted through
someone like adventure in england and the saga on how network and the phone of ship , why the beam have been also right , he had to death . nurse begin to pull a pick to lou . 24 [uni] might move to fight until onto the north officer
someone like adventure to side for the other credit , where smitty ' s unusual inside in indian recent record of a film , [uni] , his friend and being ha been the killer . when honest hand [uni] grows it but continue when only wa keeping a ransom [uni]
someone like adventure , singing , nick learns that revealing his name in the principal ' s man . however , know she fall in love with the police . beautiful , might give suicide a his college father and a arrival of the real railroad of belongs - attention
someone like adventure and understand that training soon then want to take him to intend to go by with joan and hallucinates and take a outside stage with lunch and eventually stringer gene here , while she becomes a love from labor journalist walking only and conclude that he ultimately
Epoch 8/15
1570/1569 [==============================] - 4252s 3s/step - loss: 4.1898 - accuracy: 0.3734
Dev Loss: 3.733863353729248 Dev Acc: 0.38456660509109497
someone like adventure , appears , he wa beaten by . [uni] allows hearing that her destination , keith will be shocked at money ' s house . jimmy is about that she is already chased by killed . [end]
 wounding do ,
someone like adventure and they have away him for babu , while mainly cannot steal a very extent of an figure by , he is actually the hospital , in contact with a elevator . in a mother end and reconciles with hearing that her army want a violent heart
someone like adventure to the bloody project . [end]
someone like adventure undercover . lawyer hammer thangam . morris ' s speech can see [uni] house in the house , fit a his men vikram - sacrifice . he call the killing of vote . george take place into the employee and " [uni] to be ultimately story on
someone like adventure in worse , but shoot day which us the news he kill him , ha say because " to be information to one . little village behavior . the film ha done the morrow ' s inability to live his appeal to choose for campsite in jail
someone like adventure and bid her two minion , the hospital . a : as a angel . when zach ' s behavior wa a clinic such with " combined [uni] but [uni] arrives . [end]
someone like adventure embrace or protection when he look out in his husband . country p . suddenly . sends jin by a boy , staying about the prize innocent of the village , 17 ' s murder can give further down to fire on the pretext of agreeing to
 meanwhile , the present city attempt to visit an arrow a two chief sequence . his friend concludes with the game of a boating a flip and return than bringing him
someone like adventure the ploy continues he will be the [uni] , susan , mohan , which [uni] set on to investigate [uni] , is now , who attempt to create herself . much now as after [uni] joe launch together , though lisa return unconscious . more , with
Epoch 9/15
1570/1569 [==============================] - 4262s 3s/step - loss: 4.1785 - accuracy: 0.3744
Dev Loss: 3.7286128997802734 Dev Acc: 0.38550904393196106
someone like adventure that having been set out . one of the started his throat way by it intention to directly of their horse when history . the sister temper his father [uni] and save the door , where he visible [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]
someone like adventure herself . worry , jr . ho [uni] ha a a rest of the sight ' s [uni] strip arrive and find the story to piece religion of the editor ' s memory of the story where he still take about the truth she , who is
someone like adventure next of london by the villager in the costume of marrying them , minister , is disappointed to get to agrees . on her son . the film ha not eliminate order to deal a hero . they discus all her death , but agrees to rescue
someone like adventure route , she accepts them to the police by incident and wilbur dad and cathy ' s effort for run to the upcoming vehicle at london , which meet throwing a border , he kick having to kill concrete and reiko just in australia on his sea
someone like adventure and graham . in their respect , [uni] kill daughter that make her love and carry poverty up their own - biggest room who soon finally is increasingly delayed for a group of z ' s car . a heart finished in a year - world chaos
someone like adventure in the error jam wounded to get [uni] some race , the brahmin war by wild 3 advertisement and lead him up to it with [uni] , a burton [uni] sent them by which could steal it , a train . furthermore , it can change innocent
someone like adventure and death with the truth with helping suspect pick her love with must see that his rupee a he embrace , leaving a [uni] plot on vain . thomas us a tension which came to rest of the movie . finding contact with them in manhattan .
someone like adventure and start his victim and want to one - of undercover , jake ' s object - attached after a prince . [end]
someone like adventure in his evil own break from criminal , she is charged with the world in anything association . [uni] , and keep dead when [end]
someone like adventure the remaining fight , leading the desire to convince get a kind of " fatal and unsettling ] one of a human rose who steal sworn attention . they really recognized by these old stewart . the bank seems to restore her when she stab . just
Epoch 10/15
1570/1569 [==============================] - 4234s 3s/step - loss: 4.1668 - accuracy: 0.3753
Dev Loss: 3.717137336730957 Dev Acc: 0.38603121042251587
someone like adventure with the traditional credit . louis broken up in western sacred festival , who is a dangerous shop examination , dissolute , a couple of insurance sheetal . they manage to get skillful when he say that she liked the picture of satya ' s father .
someone like adventure . [uni] [uni] and the theft among . their new blood ' s childhood daughter from his brother continues with the sake of the allied other hope is from the family leader of the murder , and gabby returned amongst [uni] , sheila accepts the american acquaintance
 [end]
someone like adventure on his phone . a professional man is given the poet manager from bob ' s and trying to hide with devotee of the epic remove heart in two - underneath taking a mine . [PAD] at the beach . hari chooses to shoot his need holding
someone like adventure - herself , and a high looking leader davis for an point a knowing remarriage sink to know his father ' s grandparent from [uni] , and several people shift out the boy by inherit , against the miss enables scientist against him . upon ellen set
someone like adventure . one of the [uni] start sex from a romantic friend and some time after sally will been , in early prince . the family explores this in the film who is forced to healing , even she push [uni] paul boat , and sam jim gordon
someone like adventure . karen will wouldn ' t be depicted without mani . dorothy leave her . this powerful [uni] then which ha since other raja , killed just after the cause of the assistant ' s lover with his injury . gopi also throw [uni] out at a
someone like adventure with the wood . when when he ' s world flee to escape love , and lunch devastated . brenda meet his - , drag winning out into a group owner prem vera how the other factory start alive , terry is disturbed by her mistress who
 in the new york accident , the general spread out in the island , marrying the money and is into [uni] . a [uni] " to showering , which ha been lot , she receives better more whereby freeing them . â [end]
someone like adventure . g . getting his singing husband of aj thing on her . pitched abruptly fall one - woman , eddie is likely living him in unscrupulous emperor from seven woman , a secret [uni] [uni] lam almost dennis , ending ted letting other companion .
Epoch 11/15
1570/1569 [==============================] - 4895s 3s/step - loss: 4.1540 - accuracy: 0.3761
Dev Loss: 3.7041451930999756 Dev Acc: 0.38710808753967285
someone like adventure tied down into those , only her life and kill blown impossible down . hilarious the window are a serious guard - bhanu is freed to help his father secretly a ranking bout . [end]
someone like adventure , including the marriage . sam himself catch outside with tower . [uni] is who , by tumor to be , and apologize to reflect the secret to carter . sweetheart is in the audience . the dispute called timber deadly screen , whose support broken go
 this in his cane , the [uni] underworld event , and good - age vessel [uni] , youth giving which his farmhouse still
someone like adventure and her the cousin are confirming to rise life . [end]
someone like adventure about renu and finding him and apologizes to [uni] , leaving dr . [uni] , whose friend show their own and ' s relative ha one , can live ' s a shore . the jewel are defeated because she have been quandary . alex and king
 , split dressed a climate bout , but with two impostor [uni] is suffering at sanity with the other freedom . karan convinces herself that [uni] happens to the ethic help who killed [uni] and focus by
someone like adventure up quietly and the judge get off from the life of the family , but produce the world station . her son [uni] provided it to be ready to show to his upcoming table . [end]
 however , sometimes continued reveals word ' s only friend and he free it to probably his arrival in customer and recruit parent a israeli spring , [uni] seems his way away . [end]
someone like adventure . ahmed change an cabin undetected . federal living day when rachel not survived and whom he is shown against a powerful week and failure . " [uni] is a rahul . she happens , and pursued in his partner out of pursuit - of outnumbered from
someone like adventure which ha initially another minute but is at the unmarried stage to directs raghuram where he promise the [uni] . he is home to the and galloway ' s house for her . the next jump from the heart of the father for the anarchist beggar using
Epoch 12/15
1570/1569 [==============================] - 4618s 3s/step - loss: 4.1465 - accuracy: 0.3772
Dev Loss: 3.695472002029419 Dev Acc: 0.38799941539764404
someone like adventure , part without his post . when they were willing to share dinner . vivek is promised but but [uni] locked from an press with evelyn . [end]
someone like adventure and he let " down [uni] up john call a police death . in a [uni] shop of [uni] . [uni] spot the moscow for dissuade television , and her business [uni] , and decides to refuse to win with horror . the killing [uni] set and
someone like adventure at both stock ammu in a gun . loaded with [uni] while [uni] tell him that her - helicopter accepts which is his mystery ' s airport left and hide the [uni] . [end]
 when the long 4 woman , there follows elope after him
someone like adventure after trick them just . in a lab battle between examines that and convinces her to go in himself . he refuse that the duck exit his sake from rivalry , however , one driver . while arriving by these first mother nagabhushanam [uni] identical to the
someone like adventure but ha already married it she is in love for it . [uni] that is discovered by he is seriously caught for them for please arena and religion in the ruin of the film and overhears walter ' s estranged family made . when he then remembers
someone like adventure the medical hotel , teaming up on thinking meeting pill . knowing that truth , rebellious denies a working ambiguous village searching up to give $ revenge , he belief " [uni] and remaining funeral . eventually from her old own boat garden that monastery pleads with
someone like adventure in jail grade and america power again , who becomes fascinated from sized , but leading to waste an confirmed jewelry to tom allowing him to operate with confirmation and " he had at school in the obsessed . as administering reading keep when he get his
someone like adventure in huge jungle in [uni] ' s behaviour for natural aspect of force developing on the the town nearly decides to tell recent couple he would be assassinated before he wa mi of other [uni] . when there is forced about nolan there . he held stabbed
someone like adventure house , the airplane win outside a first man , tossed the further invasion of hurt closely . that , [uni] ' s finger view died a prepares to ensure that quick can play up by the hand of her friend , his pistol and finishing in
Epoch 13/15
1570/1569 [==============================] - 4595s 3s/step - loss: 4.1405 - accuracy: 0.3781
Dev Loss: 3.6897571086883545 Dev Acc: 0.3890042006969452
someone like adventure , [uni] william sita . after they may love him if they harbor the case , , the world bribed , realizing that he ha to protect him . he resists mr . breaking up and fired the few actor project he ' t race , but
 the local girl event a david raid for the house ,
someone like adventure , but the year is well by pain and offer to stop being pregnant , who is sent down . when cut her failed deed and escalating [uni] [uni] . maid understands the whereabouts of her son staff [uni] . [end]
 after the time boy is cornered , including century [uni] , the sex officer - year - seeking opera rival store his lawyer invite his neighbor of a [uni] woman . however , he could catch up ,
someone like adventure ' s seat she is this in rich game of our fighter ploy , jumping through , video she know his men ' s " life attack , and neighbor , " the crime is anywhere visited by money and refuse to kill his property . her
someone like adventure so & her son from the nightclub of just paper , also partner [uni] [uni] living ' s head and come from him that so they is revealed to go . with the pool calling carl colony on michelle nuclear weapon and a laser married before his
someone like adventure in writer and a murder of australia in hot prominent cargo , who find her to decide to make interfering and shoot his life and musically go to killed the chinese hospital where they agree to keep a doctor from a most room , ramesh doe not
someone like adventure of the transition . with her death and [uni] , cabin discarded [uni] , causing the student he ' s mind for her trauma who may have planned to become the burglar island , going into the lighter , almost diane ' s squadron , not already
someone like adventure struggling , and he accuses the russian of [uni] . [end]
someone like adventure while arranges arrives . cooper writes at steve , their greatest right men and the elder later dog is shelter " alongside beyond the life of the jail and ultimately be a disguised a [uni] for time . [uni] decides to get one right to reject of
Epoch 14/15
1570/1569 [==============================] - 4758s 3s/step - loss: 4.1345 - accuracy: 0.3789
Dev Loss: 3.6883037090301514 Dev Acc: 0.38995832204818726
someone like adventure and of tom ' s retreat office to save the picture . he say that killer is impressed showcase [uni] ' s intention . [end]
 when the relationship and he is unknown other dunn [uni] marshall downhill . but gopal dy , butler start no job come away , and he went back before the base ' s activity . when unexpectedly doe a new york dixie - equipped consisting
someone like adventure , and the factory is protected with a pool ' s a back dy . kate us sue unknown to them . when devastated [uni] ? finally it coming to the sound for two city . [end]
someone like adventure , but our men end up pill , the police match . nazi is happily yu . [end]
someone like adventure by the acquittal house the door that margaret kept create a plan true . dave ! , , and upon these success . karen say his mother and she wa before her t would not survive , resulting between her radio sister of surekha , [uni] .
someone like adventure being not led to nair . [end]
someone like adventure with the local variety of town ' s talent , the funeral ' former home and she relents for her . [end]
someone like adventure in the blood ceremony . roger jump to the worst sign with the conspiracy , martha give him to drop the mansion is denied dealing . [uni] is no way go over on the an entirely machine and compliment his wife ' s concert . she also
someone like adventure , and break to exchange place to june , and paul and any child , suddenly trick the ground to make the bar she accidentally love them , his fiancee , who want to stop partying , is reunited with boxing permission is a mr . .
 ben son who try to sail to board so he went on on prison on a bit . he is
Epoch 15/15
1570/1569 [==============================] - 4616s 3s/step - loss: 4.1324 - accuracy: 0.3798
Dev Loss: 3.688047409057617 Dev Acc: 0.3906848132610321
someone like adventure . when no video attack for 60 idea of his grandfather . the rescue diamond is working in the cottage , whereupon [uni] ha gone a result with the scene room but immediately a young young people [uni] is on , so that burn the wall ha
someone like adventure and escape on his mission to a racer christian , and [uni] becomes seen when not an few moment ha , garage , just to find any married when another the two arrive with shanghai but so he get keith to johnson live on her by a
someone like adventure tackle not 6 , they member ' s to horton ? seeing the juvenile plain success . identifies one up - [uni] [uni] , and bandleader murder , the cold american mile ray is over over his flat , the freak . he return to the majority
someone like adventure . beer eventually finally informs him that slept with simon cheung child otherwise . , they are attempt to celebrity a he wa excellent neighbor , who find a all of the temple arrives and not gu , ' s flying injury , but urge her to
someone like adventure with counterattack , who now need have been to railway convent ' s death , not killed his cordelia inherits the werewolf rifle at the ferris town and resort to joyce in her - accused , intending to be on the company a heart first temporarily .
someone like adventure getting the lot and tell his true mother , to land towards action . [end]
someone like adventure , causing her died to accept resulted . upset a woman , a fifteen - mother spring the station ha invited a a s brother ' s finger , a taught the defeat . [uni] take charming knife when bell continues the sum of sumitra the lawyer
someone like adventure , which the vow herself fails s and his favorite prank sure the kung method father between his opportunity to work second unharmed and holding his name , though future officer [uni] fell in love and save a replacement to 
